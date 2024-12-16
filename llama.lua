local backend, model_path, prompt = ...
require("debug.luajit_options").SetOptimized()
local profiler = require("debug.profiler")
local get_time = require("debug.get_time")
local measure = require("debug.measure")
local gguf = require("gguf")
local Tokenizer = require("tokenizer")
local Tensor = require("tensor")
Tensor:UseComputeKernel(backend)
local Sampler = require("topp_sampler")

local function load_and_run(model_path, prompt, token_callback)
	local context_length = 512
	local temperature = 0.1
	local topp = 0.95
	local max_tokens = math.huge
	local seed = 1337
	local metadata, tensors = gguf.load(model_path)
	assert(metadata["tokenizer.ggml.model"] == "gpt2")
	assert(metadata["tokenizer.ggml.tokens"])
	assert(metadata["tokenizer.ggml.merges"])

	do
		local dim = metadata["llama.embedding_length"]
		local hidden_dim = metadata["llama.feed_forward_length"]
		local number_of_layers = metadata["llama.block_count"]
		local number_of_heads = metadata["llama.attention.head_count"]
		local number_of_key_value_heads = metadata["llama.attention.head_count_kv"] or
			metadata["llama.attention.head_count"]
		local vocabulary_size = #metadata["tokenizer.ggml.tokens"]
		local rms_norm_eps = metadata["llama.attention.layer_norm_rms_epsilon"] or 1e-5
		local head_size = dim / number_of_heads
		local sqrt_head_size = math.sqrt(head_size)
		local kv_dim = (dim * number_of_key_value_heads) / number_of_heads
		local kv_mul = number_of_heads / number_of_key_value_heads
		local rope_theta = metadata["llama.rope.freq_base"] or 10000

		local function build_rope_freqs(context_length, rope_theta)
			rope_theta = rope_theta or 10000
			assert(head_size % 2 == 0)
			local cr = {}
			local ci = {}
			local n = 1

			for pos = 0, context_length - 1 do
				for i = 0, head_size - 1, 2 do
					local freq = 1.0 / (rope_theta ^ (i / head_size))
					local val = pos * freq
					cr[n] = math.cos(val)
					ci[n] = math.sin(val)
					n = n + 1
				end
			end

			n = n - 1
			assert(context_length * (head_size / 2) == n)
			return cr, ci
		end

		local rope_freqs_real, rope_freqs_imag = build_rope_freqs(context_length, rope_theta)
		local floor = math.floor

		local function forward(state, weights, compute_logits)
			weights.token_embedding_table:CopyTo(state.token * dim, state.x, 0, dim)

			for i, layer in ipairs(weights.layers) do
				local key_cache = state.key_cache[i]
				local val_cache = state.val_cache[i]
				state.xb:RmsNormInPlace(state.x, layer.rms_att_weight, dim, rms_norm_eps)
				layer.wq:MatrixVectorMultiply(state.xb, state.q, dim, dim)
				layer.wk:MatrixVectorMultiply(state.xb, state.k, kv_dim, dim)
				layer.wv:MatrixVectorMultiply(state.xb, state.v, kv_dim, dim)

				for i = 0, dim - 1, 2 do
					local offset = 1 + (state.token_pos * (head_size / 2) + ((i % head_size) / 2))
					local fcr = rope_freqs_real[offset]
					local fci = rope_freqs_imag[offset]
					local rotn = i < kv_dim and 2 or 1

					for v = 0, rotn - 1 do
						local vec = v == 0 and state.q or state.k
						local v0 = vec:GetFloat(i)
						local v1 = vec:GetFloat(i + 1)
						vec:SetFloat(i, v0 * fcr - v1 * fci)
						vec:SetFloat(i + 1, v0 * fci + v1 * fcr)
					end
				end

				state.k:CopyTo(0, key_cache, state.token_pos * kv_dim, kv_dim)
				state.v:CopyTo(0, val_cache, state.token_pos * kv_dim, kv_dim)

				if not compute_logits and i == #weights.layers then
					return
				end

				for h = 0, number_of_heads - 1 do
					local qOffset = h * head_size
					local attOffset = h * context_length
					local key_cache_offset = floor(h / kv_mul) * head_size

					for t = 0, state.token_pos do
						state.att:SetFloat(
							attOffset + t,
							state.q:Dot(qOffset, key_cache, t * kv_dim + key_cache_offset, head_size) / sqrt_head_size
						)
					end

					state.att:SoftMaxInPlace(attOffset, state.token_pos + 1)
					state.xb:FillInPlace(qOffset, head_size, 0)

					for t = 0, state.token_pos do
						state.xb:SaxpyInPlace(
							qOffset,
							val_cache,
							t * kv_dim + key_cache_offset,
							head_size,
							state.att:GetFloat(attOffset + t)
						)
					end
				end

				layer.wo:MatrixVectorMultiply(state.xb, state.xb2, dim, dim)
				state.x:AddTensorInPlace(state.xb2)
				state.xb:RmsNormInPlace(state.x, layer.rms_ffn_weight, dim, rms_norm_eps)
				layer.w1:MatrixVectorMultiply(state.xb, state.hb, hidden_dim, dim)
				layer.w3:MatrixVectorMultiply(state.xb, state.hb2, hidden_dim, dim)
				state.hb:SigmoidInPlace()
				state.hb:MultiplyTensorInPlace(state.hb2)
				layer.w2:MatrixVectorMultiply(state.hb, state.xb, dim, hidden_dim)
				state.x:AddTensorInPlace(state.xb)
			end

			state.x:RmsNormInPlace(state.x, weights.rms_final_weight, dim, rms_norm_eps)
			weights.wcls:MatrixVectorMultiply(state.x, state.logits, vocabulary_size, dim)
		end

		local tokenizer = Tokenizer:new(metadata["tokenizer.ggml.tokens"], metadata["tokenizer.ggml.merges"])
		local sampler = Sampler:new(vocabulary_size, temperature, topp)
		local prompt_tokens = tokenizer:EncodeString(prompt)
		local weights = {layers = {}}

		do
			local function load_tensor(name)
				local entry = tensors[name]

				if Tensor[entry.type_info.name] then
					return Tensor[entry.type_info.name](Tensor, entry.size, entry.blob):SetName(name .. "[" .. entry.type_info.name .. "]")
				end

				error("NYI tensor type: " .. entry.type_info.name)
			end

			weights.token_embedding_table = load_tensor("token_embd.weight")

			for i = 0, number_of_layers - 1 do
				weights.layers[i + 1] = {
					rms_att_weight = load_tensor("blk." .. i .. ".attn_norm.weight"),
					wq = load_tensor("blk." .. i .. ".attn_q.weight"),
					wk = load_tensor("blk." .. i .. ".attn_k.weight"),
					wv = load_tensor("blk." .. i .. ".attn_v.weight"),
					wo = load_tensor("blk." .. i .. ".attn_output.weight"),
					rms_ffn_weight = load_tensor("blk." .. i .. ".ffn_norm.weight"),
					w1 = load_tensor("blk." .. i .. ".ffn_gate.weight"),
					w2 = load_tensor("blk." .. i .. ".ffn_down.weight"),
					w3 = load_tensor("blk." .. i .. ".ffn_up.weight"),
				}
			end

			weights.rms_final_weight = load_tensor("output_norm.weight")
			weights.wcls = load_tensor("output.weight")
		end

		local state = {
			token = tokenizer:EncodeString("<|begin_of_text|>")[1] - 1,
			token_pos = 0,
			x = Tensor:F32(dim),
			xb = Tensor:F32(dim),
			xb2 = Tensor:F32(dim),
			hb = Tensor:F32(hidden_dim),
			hb2 = Tensor:F32(hidden_dim),
			q = Tensor:F32(dim),
			k = Tensor:F32(dim),
			v = Tensor:F32(dim),
			att = Tensor:F32(number_of_heads * context_length),
			logits = Tensor:F32(vocabulary_size),
			key_cache = {},
			val_cache = {},
		}

		for i = 1, number_of_layers do
			state.key_cache[i] = Tensor:F32(context_length * kv_dim)
			state.val_cache[i] = Tensor:F32(context_length * kv_dim)
		end

		if backend == "cuda" then
			-- upload and preallocate tensor memory for better performance and vram usage
			local total_size = 0
			local gpu = require("compute.gpu_cuda")
			measure("uploading tensors to gpu")
			local size_map = {}

			for _, tensor in ipairs(Tensor.GetAll()) do
				if tensor.name and tensor.name:find(".weight") then
					-- weight tensors are static
					tensor.blob.gpu_ptr = gpu.allocate_on_device(tensor.blob.byte_size, tensor.blob.blob)
					total_size = total_size + tensor.blob.byte_size
				else
					-- state tensors are dynamic and are uploaded on each Tensor.MatrixVectorMultiply call
					-- so we can allocate and share memory for each byte size
					size_map[tensor.blob.byte_size] = size_map[tensor.blob.byte_size] or {}
					table.insert(size_map[tensor.blob.byte_size], tensor)
				end
			end

			for byte_size, tensors in pairs(size_map) do
				local gpu_ptr = gpu.allocate_on_device(byte_size)

				for _, tensor in ipairs(tensors) do
					tensor.blob.gpu_ptr = gpu_ptr
				end

				total_size = total_size + byte_size
			end

			measure()
			print(string.format("%.2fgb tensors allocated on GPU", total_size / 1024 / 1024 / 1024))
			gpu.dump_gpu_stats()
		end

		do
			local total_size = 0

			for _, tensor in ipairs(Tensor.GetAll()) do
				total_size = total_size + tensor.blob.byte_size
			end

			print(string.format("%.2fgb tensors allocated on CPU", total_size / 1024 / 1024 / 1024))
		end

		profiler.Start()
		local total_time = 0
		print("\n\n\n")
		math.randomseed(seed)

		while state.token_pos < max_tokens do
			local start_time = get_time()
			local next_token
			
			if state.token_pos < #prompt_tokens then
				forward(state, weights, false)
				next_token = prompt_tokens[state.token_pos + 1]
			else
				forward(state, weights, true)
				next_token = sampler:SampleToken(state.logits) + 1
			end

			state.token = next_token - 1

			do
				local token_string = tokenizer:TokenToString(state.token + 1)
				token_callback(token_string)

				if state.token_pos >= #prompt_tokens and token_string == "<|eot_id|>" then
					break
				end
			end

			state.token_pos = state.token_pos + 1
			total_time = total_time + get_time() - start_time
		end

		print("\n\n\n")

		if backend == "cuda" then require("compute.gpu_cuda").dump_gpu_stats() end

		profiler.Stop()
		local token_count = (state.token_pos + 1)
		local tokens_per_sec = 1 / (total_time / token_count)
		print(
			string.format(
				"token count: %i\nelapsed: %.2fs\n%.2f tokens/s",
				token_count,
				total_time,
				tokens_per_sec
			)
		)

		if backend == "cuda" and false then -- LOL
			local gpu = require("compute.gpu_cuda")
			local done = {}

			for _, tensor in ipairs(Tensor.GetAll()) do
				if not done[tensor.blob.gpu_ptr] then
					gpu.free_on_device(tensor.blob.gpu_ptr)
					done[tensor.blob.gpu_ptr] = true
				end
			end
		end
	end
end

load_and_run(model_path, "<|start_header_id|>user<|end_header_id|>\n" .. prompt .. "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n", function(token)
	io.write(token)
	io.flush()
end)