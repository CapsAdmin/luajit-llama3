local Tensor = require("tensor")
local ffi = require("ffi")

local function loadTensor(entry, name)
	if Tensor[entry.type_info.name] then
		local t = Tensor[entry.type_info.name](Tensor, entry.size, entry.blob):SetName(name .. "[" .. entry.type_info.name .. "]")

		if name == "token_embd.weight" then
			local test = {
				13.427734,
				53.710938,
				-26.855469,
				-0.0,
				-26.855469,
				-0.0,
				-13.427734,
				-13.427734,
				-26.855469,
				-26.855469,
				-13.427734,
				-93.99414,
				26.855469,
				13.427734,
				-80.56641,
				80.56641,
				-53.710938,
				-40.283203,
				-67.13867,
				-0.0,
				13.427734,
				107.421875,
				-13.427734,
				-26.855469,
				-26.855469,
				-13.427734,
				-0.0,
				40.283203,
				40.283203,
				13.427734,
				-80.56641,
				-13.427734,
				-69.885254,
				-69.885254,
				-0.0,
				-0.0,
				-104.82788,
				-34.942627,
				-139.77051,
				-0.0,
				69.885254,
				-69.885254,
				-0.0,
				34.942627,
				34.942627,
				-34.942627,
				-0.0,
				279.54102,
				-0.0,
				-0.0,
				34.942627,
				34.942627,
				-34.942627,
				-34.942627,
				-0.0,
				-0.0,
				-34.942627,
				-34.942627,
				-139.77051,
				-0.0,
				-0.0,
				-34.942627,
				69.885254,
				-104.82788,
				-20.141602,
				20.141602,
				-0.0,
				-80.56641,
				-60.424805,
				-60.424805,
				20.141602,
				-20.141602,
				40.283203,
				40.283203,
				-20.141602,
				20.141602,
				-0.0,
				-60.424805,
				-60.424805,
				120.84961,
				-20.141602,
				-40.283203,
				-40.283203,
				-80.56641,
				-0.0,
				-20.141602,
				60.424805,
				161.13281,
				-0.0,
				-120.84961,
				-40.283203,
				-0.0,
				-20.141602,
				140.99121,
				60.424805,
				-0.0,
				-104.98047,
				-26.245117,
				26.245117,
				91.85791,
				39.367676,
			}
			local errored = false
			local str = ""

			for i, v in ipairs(test) do
				if i == 65 then break end

				v = math.floor(v)
				local our = math.floor(t:GetFloat(i - 1) * 10000)

				if our ~= v then
					str = str .. (i - 1 .. ": " .. our .. " ~= " .. v) .. "\n"
					errored = true
				end
			end

			if errored then
				print(str)
				error("quantized reading is wrong")
			end
		end

		return t
	end

	error("NYI tensor type: " .. entry.type_info.name)
end

local function loadTensorArray(size, getTensorEntry)
	local array = {}

	for i = 1, size do
		array[i] = loadTensor(getTensorEntry(i-1))
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
		-- weights for MatrixDotProducts
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