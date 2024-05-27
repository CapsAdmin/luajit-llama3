local ffi = require("ffi")
local ggf = require("gguf")
local Tokenizer = require("tokenizer")
local Configuration = require("configuration")
local Weights = require("weights")
local Sampler = require("sampler")
local Tensor = require("tensor")
local gguf = ggf.load_gguf("/home/caps/projects/llama3.java/Meta-Llama-3-8B-Instruct-Q4_0.gguf")
assert(gguf.metadata["tokenizer.ggml.model"] == "gpt2")
assert(gguf.metadata["tokenizer.ggml.tokens"])
assert(gguf.metadata["tokenizer.ggml.merges"])
local tokenizer = Tokenizer(gguf.metadata["tokenizer.ggml.tokens"], gguf.metadata["tokenizer.ggml.merges"])
local configuration = Configuration(1000, gguf.metadata, tokenizer.vocabulary.size())
local weights = Weights(gguf.tensors, configuration.numberOfLayers)
local sampler = Sampler(tokenizer.vocabulary.size(), 0, 0.95, 1337)
local context = [[<|start_header_id|>system<|end_header_id|>
]] .. "you are a helpful chatbot" .. [[<|eot_id|><|start_header_id|>user<|end_header_id|>
]] .. "hello" .. [[<|eot_id|><|start_header_id|>assistant<|end_header_id|><|eot_id|>]]

for k, v in ipairs(tokenizer.encode(context)) do
	print(k, v, tokenizer.vocabulary.tokens[v])
end

local function rmsnorm(out, x, weight, size, rmsNormEps)
	local ss = x:Reduce(0, size, 0, function(acc, xi) return acc + xi * xi end);
	ss = ss / size;
	ss = ss + rmsNormEps;
	ss = 1.0 / math.sqrt(ss)
    print("map in place")
	out:MapWithIndexInPlace(0, size, function(value, index) 
		return weight:GetFloat(index) * (ss * x:GetFloat(index)) 
	end)
end

local function forward(c, w, s, token, position)
	local dim = c.dim
	local headSize = c.headSize
	local kvDim = (c.dim * c.numberOfKeyValueHeads) / c.numberOfHeads
	local kvMul = c.numberOfHeads / c.numberOfKeyValueHeads
	local sqrtHeadSize = math.sqrt(headSize)
	w.token_embedding_table:CopyTo(token * dim, s.x, 0, dim)

	for l = 1, c.numberOfLayers do
		print("layer: ", l)
		rmsnorm(s.xb, s.x, w.rms_att_weight[l], dim, c.rmsNormEps)
		w.wq[l]:MatMul(s.xb, s.q, dim, dim)
		error("UH OH")
		w.wk[l]:MatMul(s.xb, s.k, kvDim, dim)
		w.wv[l]:MatMul(s.xb, s.v, kvDim, dim)

		error("HUH HOH")

		for i = 0, dim - 1, 2 do
			local head_dim = i % headSize
			local fcr = w.freq_cis_real:Get(position * (headSize / 2) + (head_dim / 2))
			local fci = w.freq_cis_imag:Get(position * (headSize / 2) + (head_dim / 2))
			local rotn = i < kvDim and 2 or 1

			for v = 0, rotn - 1 do
				local vec = v == 0 and s.q or s.v
				local v0 = vec:GetFloat(i)
				local v1 = vec:GetFloat(i + 1)
				vec:SetFloat(i, v0 * fcr - v1 * fci)
				vec:SetFloat(i + 1, v0 * fcr + v1 * fci)
			end

			s.k:copyTo(0, s.keyCache[l], position * kvDim, kvDim)
			s.v:copyTo(0, s.valueCache[l], position * kvDim, kvDim)
			local curLayer = l

			for h = 0, c.numberOfHeads do
				local qOffset = h * headSize
				local attOffset = h * c.contextLength

				for t = 0, position do
					local keyCacheOffset = t * kvDim + (h / kvMul) * headSize
					local score = s.q:Dot(qOffset, s.keyCache[curLayer], keyCacheOffset, headSize)
					score = score / sqrtHeadSize
					s.att:SetFloat(attOffset + t, score)
				end

				s.att:SoftMaxInPlace(attOffset, position + 1)
				local xbOffset = h * headSize
				s.xb:FillInPlace(xbOffset, headSize, 0)

				for t = 0, position do
					local vOffset = t * kvDim + (h / kvMul) * headSize
					local a = state.att:GetFloat(attOffset + t)
					s.xb:SaxyInPlace(xbOffset, s.valueCache[curLayer], vOffset, headSize, a)
				end
			end
		end

		w.wo[l]:MatMul(s.xb, s.xb2, dim, dim)
		s.x:addInPlace(s.xb2)
		rmsnorm(s.xb, s.x, w.rms_ffn_weight[l], dim, c.rmsNormEps)
		w.w1[l]:MatMul(s.xb, s.hb, c.hiddenDim, dim)
		w.w3[l]:MatMul(s.xb, s.hb2, c.hiddenDim, dim)

		s.hb:mapInPlace(function(value)
			return value / (1.0 + math.exp(-value))
		end)

		s.hb:multiplyInPlace(s.hb2)
		w.w2[l]:MatMul(s.hb, s.xb, dim, c.hiddenDim)
		s.x:addInPlace(s.xb)
	end
end

local config = configuration

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
	keyCache = {},
	valueCache = {},
	latestToken = 0,
}
local token = 0
local pos = 0
forward(configuration, weights, state, token, pos)