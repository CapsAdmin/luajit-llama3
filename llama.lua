local model_path = ...
require("luajit_options")()

do
	local timers = {}

	function timer(what)
		if what then
			table.insert(timers, 1, {what = what, time = os.clock()})
		else
			local t = table.remove(timers, 1)
			print(t.what .. " took " .. (os.clock() - t.time) .. " seconds")
		end
	end
end

local ffi = require("ffi")
local ggf = require("gguf")
local Tokenizer = require("tokenizer")
local Configuration = require("configuration")
local Weights = require("weights")
local Tensor = require("tensor")
local gguf = ggf.load_gguf(model_path)
local Sampler = require("topp_sampler")
assert(gguf.metadata["tokenizer.ggml.model"] == "gpt2")
assert(gguf.metadata["tokenizer.ggml.tokens"])
assert(gguf.metadata["tokenizer.ggml.merges"])
local tokenizer = Tokenizer(gguf.metadata["tokenizer.ggml.tokens"], gguf.metadata["tokenizer.ggml.merges"])
local configuration = Configuration(8192, gguf.metadata, tokenizer.vocabulary.size())
local weights = Weights(gguf.tensors, configuration.numberOfLayers)

pcall(function()
	local inject_threaded_matmul = require("threads")
	inject_threaded_matmul(Tensor)
end)

local function rmsnorm(out, x, weight, size, rmsNormEps)
	local ss = x:Reduce(0, size, 0, function(acc, xi)
		return acc + xi * xi
	end)
	ss = ss / size
	ss = ss + rmsNormEps
	ss = 1.0 / math.sqrt(ss)

	out:MapInPlace(0, size, function(value, index)
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

	for l = 0, c.numberOfLayers - 1 do
		rmsnorm(s.xb, s.x, w.rms_att_weight[l], dim, c.rmsNormEps)
		w.wq[l]:MatMul(s.xb, s.q, dim, dim)
		w.wk[l]:MatMul(s.xb, s.k, kvDim, dim)
		w.wv[l]:MatMul(s.xb, s.v, kvDim, dim)

		for i = 0, dim - 1, 2 do
			local head_dim = i % headSize
			local fcr = c.ropeFreqsReal[1 + ((position * (headSize / 2) + (head_dim / 2)))]
			local fci = c.ropeFreqsImag[1 + ((position * (headSize / 2) + (head_dim / 2)))]
			local rotn = i < kvDim and 2 or 1

			for v = 0, rotn - 1 do
				local vec = v == 0 and s.q or s.k
				local v0 = vec:GetFloat(i)
				local v1 = vec:GetFloat(i + 1)
				vec:SetFloat(i, v0 * fcr - v1 * fci)
				vec:SetFloat(i + 1, v0 * fci + v1 * fcr)
			end
		end

		s.k:CopyTo(0, s.keyCache[l + 1], position * kvDim, kvDim)
		s.v:CopyTo(0, s.valueCache[l + 1], position * kvDim, kvDim)
		local curLayer = l

		for h = 0, c.numberOfHeads - 1 do
			local qOffset = h * headSize
			local attOffset = h * c.contextLength

			for t = 0, position do
				local keyCacheOffset = t * kvDim + math.floor(h / kvMul) * headSize
				local score = s.q:Dot(qOffset, s.keyCache[curLayer + 1], keyCacheOffset, headSize)
				score = score / sqrtHeadSize
				s.att:SetFloat(attOffset + t, score)
			end

			s.att:SoftMaxInPlace(attOffset, position + 1)
			local xbOffset = h * headSize
			s.xb:FillInPlace(xbOffset, headSize, 0)

			for t = 0, position do
				local vOffset = t * kvDim + math.floor(h / kvMul) * headSize
				local a = s.att:GetFloat(attOffset + t)
				s.xb:SaxyInPlace(xbOffset, s.valueCache[curLayer + 1], vOffset, headSize, a)
			end
		end

		w.wo[l]:MatMul(s.xb, s.xb2, dim, dim)
		s.x:AddTensorInPlace(s.xb2)
		rmsnorm(s.xb, s.x, w.rms_ffn_weight[l], dim, c.rmsNormEps)
		w.w1[l]:MatMul(s.xb, s.hb, c.hiddenDim, dim)
		w.w3[l]:MatMul(s.xb, s.hb2, c.hiddenDim, dim)

		s.hb:MapInPlace(0, s.hb.size, function(value)
			return value / (1.0 + math.exp(-value))
		end)

		s.hb:MultiplyTensorInPlace(s.hb2)
		w.w2[l]:MatMul(s.hb, s.xb, dim, c.hiddenDim)
		s.x:AddTensorInPlace(s.xb)
	end

	rmsnorm(s.x, s.x, w.rms_final_weight, dim, c.rmsNormEps)
	w.wcls:MatMul(s.x, s.logits, c.vocabularySize, dim)
end

local config = configuration
local kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads
local state = {
	x = Tensor:F32(config.dim):SetName("x[F32]"),
	xb = Tensor:F32(config.dim):SetName("xb[F32]"),
	xb2 = Tensor:F32(config.dim):SetName("xb2[F32]"),
	hb = Tensor:F32(config.hiddenDim):SetName("hb[F32]"),
	hb2 = Tensor:F32(config.hiddenDim):SetName("hb2[F32]"),
	q = Tensor:F32(config.dim):SetName("q[F32]"),
	k = Tensor:F32(config.dim):SetName("k[F32]"),
	v = Tensor:F32(config.dim):SetName("v[F32]"),
	att = Tensor:F32(config.numberOfHeads * config.contextLength):SetName("att[F32]"),
	logits = Tensor:F32(config.vocabularySize):SetName("logits[F32]"),
	keyCache = (function()
		local a = {}

		for i = 1, config.numberOfLayers do
			a[i] = Tensor:F32(config.contextLength * kvDim)
		end

		return a
	end)(),
	valueCache = (function()
		local a = {}

		for i = 1, config.numberOfLayers do
			a[i] = Tensor:F32(config.contextLength * kvDim)
		end

		return a
	end)(),
}
_G.gc_refs = {
	state,
	weights,
}

local sampler = Sampler:new(config.vocabularySize, 0.1, 0.95)
local context = [[<|start_header_id|>user<|end_header_id|>
hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>
]]
local tokens = tokenizer.encode(context)
local pos = 0
local token = tokenizer.encode("<|begin_of_text|>")[1]
local next_token

for pos = 1, 30 do
	forward(configuration, weights, state, token - 1, pos - 1)

	if pos < #tokens then
		next_token = tokens[pos]
	else
		next_token = sampler:SampleToken(state.logits)+1
	end

	token = next_token
	print(tokenizer.vocabulary.tokens[token])
end