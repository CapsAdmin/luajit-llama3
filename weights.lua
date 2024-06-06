local Tensor = require("tensor")
local ffi = require("ffi")

local function loadTensor(entry, name)
	if Tensor[entry.type_info.name] then
		return Tensor[entry.type_info.name](Tensor, entry.size, entry.blob):SetName(name .. "[" .. entry.type_info.name .. "]")
	end

	error("NYI tensor type: " .. entry.type_info.name)
end

local function loadTensorArray(size, getTensorEntry)
	local array = {}

	for i = 1, size do
		array[i] = loadTensor(getTensorEntry(i - 1))
	end

	return array
end

local function Weights(tensors, numberOfLayers)
	return {
		-- token embedding table
		token_embedding_table = loadTensor(tensors["token_embd.weight"], "token_embd.weight"), -- (vocab_size, dim)
		-- weights for rmsnorms
		rms_att_weight = loadTensorArray(numberOfLayers, function(i)
			local name = "blk." .. i .. ".attn_norm.weight"
			return tensors[name], name
		end), -- (layer, dim) rmsnorm weights
		-- weights for MatrixVectorMultiplys
		wq = loadTensorArray(numberOfLayers, function(i)
			local name = "blk." .. i .. ".attn_q.weight"
			return tensors[name], name
		end), -- (layer, n_heads * head_size)
		wk = loadTensorArray(numberOfLayers, function(i)
			local name = "blk." .. i .. ".attn_k.weight"
			return tensors[name], name
		end), -- (layer, n_kv_heads, head_size)
		wv = loadTensorArray(numberOfLayers, function(i)
			local name = "blk." .. i .. ".attn_v.weight"
			return tensors[name], name
		end), -- (layer, n_kv_heads * head_size)
		wo = loadTensorArray(numberOfLayers, function(i)
			local name = "blk." .. i .. ".attn_output.weight"
			return tensors[name], name
		end), -- (layer, n_heads * head_size, dim)
		rms_ffn_weight = loadTensorArray(numberOfLayers, function(i)
			local name = "blk." .. i .. ".ffn_norm.weight"
			return tensors[name], name
		end), -- (layer, dim)
		-- weights for ffn
		w1 = loadTensorArray(numberOfLayers, function(i)
			local name = "blk." .. i .. ".ffn_gate.weight"
			return tensors[name], name
		end), -- w1 -- (layer, hidden_dim, dim)
		w2 = loadTensorArray(numberOfLayers, function(i)
			local name = "blk." .. i .. ".ffn_down.weight"
			return tensors[name], name
		end), -- w2 -- (layer, dim, hidden_dim)
		w3 = loadTensorArray(numberOfLayers, function(i)
			local name = "blk." .. i .. ".ffn_up.weight"
			return tensors[name], name
		end), -- w -- (layer, hidden_dim, dim)
		-- public final rmsnorm
		rms_final_weight = loadTensor(tensors["output_norm.weight"], "output_norm.weight"), -- (dim,)
		-- (optional) classifier weights for the logits, on the last layer
		wcls = loadTensor(tensors["output.weight"], "output.weight"), -- (vocab_size, dim)
	}
end

return Weights