local backend, model_path, prompt = ...
require("debug.luajit_options")()
local profiler = require("debug.profiler")
_G.measure = require("debug.measure")
local gguf = require("gguf")
local Tokenizer = require("tokenizer")
local Configuration = require("configuration")
local Weights = require("weights")
local Tensor = require("tensor")
local Sampler = require("topp_sampler")
if backend ~= "lua" then
	require("tensor_compute_ext")["use_" .. backend]()
end

local function load_and_run(model_path, prompt, token_callback)
	local context_length = 512
	local temperature = 0.1
	local topp = 0.95
	local max_tokens = math.huge
	
	local math_exp = math.exp
	local floor = math.floor

	local metadata, tensors = gguf.load(model_path)
	assert(metadata["tokenizer.ggml.model"] == "gpt2")
	assert(metadata["tokenizer.ggml.tokens"])
	assert(metadata["tokenizer.ggml.merges"])
	local tokenizer = Tokenizer(metadata["tokenizer.ggml.tokens"], metadata["tokenizer.ggml.merges"])
	local config = Configuration(context_length, metadata, tokenizer.vocabulary.size())
	local weights = Weights(tensors, config.numberOfLayers)
	local kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads
	local sampler = Sampler:new(config.vocabularySize, temperature, topp)

	local tokens = tokenizer.encode(prompt)

	local function kv_cache()
		local a = {}

		for i = 1, config.numberOfLayers do
			a[i] = Tensor:F32(config.contextLength * kvDim)
		end

		return a
	end

	local state = {
		x = Tensor:F32(config.dim),
		xb = Tensor:F32(config.dim),
		xb2 = Tensor:F32(config.dim),
		hb = Tensor:F32(config.hiddenDim),
		hb2 = Tensor:F32(config.hiddenDim),
		q = Tensor:F32(config.dim),
		k = Tensor:F32(config.dim),
		v = Tensor:F32(config.dim),
		att = Tensor:F32(config.numberOfHeads * config.contextLength),
		logits = Tensor:F32(config.vocabularySize),
		keyCache = kv_cache(),
		valueCache = kv_cache(),
	}
	

	local function exp(value)
		return value / (1.0 + math_exp(-value))
	end

	local function forward(token, position)
		local dim = config.dim
		local headSize = config.headSize
		local kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads
		local kvMul = config.numberOfHeads / config.numberOfKeyValueHeads
		local sqrtHeadSize = math.sqrt(headSize)
		weights.token_embedding_table:CopyTo(token * dim, state.x, 0, dim)

		for l = 1, config.numberOfLayers do
			local keyCache = state.keyCache[l]
			local valueCache = state.valueCache[l]

			state.xb:RmsNormInPlace(state.x, weights.rms_att_weight[l], dim, config.rmsNormEps)
			weights.wq[l]:MatrixVectorMultiply(state.xb, state.q, dim, dim)
			weights.wk[l]:MatrixVectorMultiply(state.xb, state.k, kvDim, dim)
			weights.wv[l]:MatrixVectorMultiply(state.xb, state.v, kvDim, dim)

			for i = 0, dim - 1, 2 do
				local head_dim = i % headSize
				local fcr = config.ropeFreqsReal[1 + ((position * (headSize / 2) + (head_dim / 2)))]
				local fci = config.ropeFreqsImag[1 + ((position * (headSize / 2) + (head_dim / 2)))]
				local rotn = i < kvDim and 2 or 1

				for v = 0, rotn - 1 do
					local vec = v == 0 and state.q or state.k
					local v0 = vec:GetFloat(i)
					local v1 = vec:GetFloat(i + 1)
					vec:SetFloat(i, v0 * fcr - v1 * fci)
					vec:SetFloat(i + 1, v0 * fci + v1 * fcr)
				end
			end

			state.k:CopyTo(0, keyCache, position * kvDim, kvDim)
			state.v:CopyTo(0, valueCache, position * kvDim, kvDim)

			for h = 0, config.numberOfHeads - 1 do
				local qOffset = h * headSize
				local attOffset = h * config.contextLength

				for t = 0, position do
					local keyCacheOffset = t * kvDim + floor(h / kvMul) * headSize
					local score = state.q:Dot(qOffset, keyCache, keyCacheOffset, headSize)
					score = score / sqrtHeadSize
					state.att:SetFloat(attOffset + t, score)
				end

				state.att:SoftMaxInPlace(attOffset, position + 1)
				local xbOffset = h * headSize
				state.xb:FillInPlace(xbOffset, headSize, 0)

				for t = 0, position do
					local vOffset = t * kvDim + floor(h / kvMul) * headSize
					local a = state.att:GetFloat(attOffset + t)
					state.xb:SaxyInPlace(xbOffset, valueCache, vOffset, headSize, a)
				end
			end

			weights.wo[l]:MatrixVectorMultiply(state.xb, state.xb2, dim, dim)
			state.x:AddTensorInPlace(state.xb2)
			state.xb:RmsNormInPlace(state.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps)
			weights.w1[l]:MatrixVectorMultiply(state.xb, state.hb, config.hiddenDim, dim)
			weights.w3[l]:MatrixVectorMultiply(state.xb, state.hb2, config.hiddenDim, dim)
			state.hb:MapInPlace(0, state.hb.size, exp)
			state.hb:MultiplyTensorInPlace(state.hb2)
			weights.w2[l]:MatrixVectorMultiply(state.hb, state.xb, dim, config.hiddenDim)
			state.x:AddTensorInPlace(state.xb)
		end

		state.x:RmsNormInPlace(state.x, weights.rms_final_weight, dim, config.rmsNormEps)
		weights.wcls:MatrixVectorMultiply(state.x, state.logits, config.vocabularySize, dim)
	end

	local token = tokenizer.encode("<|begin_of_text|>")[1]
	local next_token

	for pos = 1, max_tokens do
		forward(token - 1, pos - 1)

		if pos < #tokens then
			next_token = tokens[pos]
		else
			next_token = sampler:SampleToken(state.logits) + 1
		end

		token = next_token
		local token_string = tokenizer.token_to_string(token)
		if pos > #tokens and token_string == "<|eot_id|>" then
			break
		end
		token_callback(token_string)
	end
end

local prompt = [[<|start_header_id|>user<|end_header_id|>
]]..prompt..[[<|eot_id|><|start_header_id|>assistant<|end_header_id|>
]]

load_and_run(model_path, prompt, function(token)
	io.write(token)
	io.flush()
end)