local backend, model_path, prompt = ...
require("debug.luajit_options")()
local profiler = require("debug.profiler")
local get_time = require("debug.get_time")
local measure = require("debug.measure")
local gguf = require("gguf")
local Tokenizer = require("tokenizer")
local Weights = require("weights")
local Tensor = require("tensor")
local Sampler = require("topp_sampler")

if backend ~= "lua" then require("tensor_compute_ext")["use_" .. backend]() end

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
		local number_of_key_value_heads = metadata["llama.attention.head_count_kv"] and
			metadata["llama.attention.head_count_kv"] or
			metadata["llama.attention.head_count"]
		local vocabulary_size = #metadata["tokenizer.ggml.tokens"]
		local rms_norm_eps = metadata["llama.attention.layer_norm_rms_epsilon"] or 1e-5
		local head_size = dim / number_of_heads
		local kv_dim = (dim * number_of_key_value_heads) / number_of_heads
		local kv_mul = number_of_heads / number_of_key_value_heads
		local sqrt_head_size = math.sqrt(head_size)
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

		local function forward(state, weights)
			weights.token_embedding_table:CopyTo(state.token * dim, state.x, 0, dim)

			for l = 1, number_of_layers do
				local key_cache = state.key_cache[l]
				local val_cache = state.val_cache[l]
				state.xb:RmsNormInPlace(state.x, weights.rms_att_weight[l], dim, rms_norm_eps)
				weights.wq[l]:MatrixVectorMultiply(state.xb, state.q, dim, dim)
				weights.wk[l]:MatrixVectorMultiply(state.xb, state.k, kv_dim, dim)
				weights.wv[l]:MatrixVectorMultiply(state.xb, state.v, kv_dim, dim)

				for i = 0, dim - 1, 2 do
					local head_dim = i % head_size
					local fcr = rope_freqs_real[1 + ((state.token_pos * (head_size / 2) + (head_dim / 2)))]
					local fci = rope_freqs_imag[1 + ((state.token_pos * (head_size / 2) + (head_dim / 2)))]
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

				for h = 0, number_of_heads - 1 do
					local qOffset = h * head_size
					local attOffset = h * context_length

					for t = 0, state.token_pos do
						local key_cache_offset = t * kv_dim + floor(h / kv_mul) * head_size
						local score = state.q:Dot(qOffset, key_cache, key_cache_offset, head_size)
						score = score / sqrt_head_size
						state.att:SetFloat(attOffset + t, score)
					end

					state.att:SoftMaxInPlace(attOffset, state.token_pos + 1)
					local xbOffset = h * head_size
					state.xb:FillInPlace(xbOffset, head_size, 0)

					for t = 0, state.token_pos do
						local vOffset = t * kv_dim + floor(h / kv_mul) * head_size
						local a = state.att:GetFloat(attOffset + t)
						state.xb:SaxpyInPlace(xbOffset, val_cache, vOffset, head_size, a)
					end
				end

				weights.wo[l]:MatrixVectorMultiply(state.xb, state.xb2, dim, dim)
				state.x:AddTensorInPlace(state.xb2)
				state.xb:RmsNormInPlace(state.x, weights.rms_ffn_weight[l], dim, rms_norm_eps)
				weights.w1[l]:MatrixVectorMultiply(state.xb, state.hb, hidden_dim, dim)
				weights.w3[l]:MatrixVectorMultiply(state.xb, state.hb2, hidden_dim, dim)
				state.hb:SigmoidInPlace()
				state.hb:MultiplyTensorInPlace(state.hb2)
				weights.w2[l]:MatrixVectorMultiply(state.hb, state.xb, dim, hidden_dim)
				state.x:AddTensorInPlace(state.xb)
			end

			state.x:RmsNormInPlace(state.x, weights.rms_final_weight, dim, rms_norm_eps)
			weights.wcls:MatrixVectorMultiply(state.x, state.logits, vocabulary_size, dim)
		end

		local tokenizer = Tokenizer:new(metadata["tokenizer.ggml.tokens"], metadata["tokenizer.ggml.merges"])
		local sampler = Sampler:new(vocabulary_size, temperature, topp)
		local tokens = tokenizer:EncodeString(prompt)
		local weights = Weights(tensors, number_of_layers)
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
					gpu.copy_to_device(gpu_ptr, tensor.blob.blob, byte_size)
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

		--profiler.Start()
		local total_time = 0
		print("\n\n\n")
		math.randomseed(seed)

		while state.token_pos < max_tokens do
			local start_time = get_time()
			forward(state, weights)
			local next_token

			if state.token_pos < #tokens then
				next_token = tokens[state.token_pos + 1]
			else
				next_token = sampler:SampleToken(state.logits) + 1
			end

			state.token = next_token - 1

			do
				local token_string = tokenizer:TokenToString(state.token + 1)
				token_callback(token_string)

				if state.token_pos >= #tokens and token_string == "<|eot_id|>" then break end
			end

			state.token_pos = state.token_pos + 1
			total_time = total_time + get_time() - start_time
		end

		print("\n\n\n")

		if backend == "cuda" then require("compute.gpu_cuda").dump_gpu_stats() end

		--profiler.Stop()
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

local prompt = [[<|start_header_id|>user<|end_header_id|>
]] .. prompt .. [[<|eot_id|><|start_header_id|>assistant<|end_header_id|>
]]

load_and_run(model_path, prompt, function(token)
	io.write(token)
	io.flush()
end)