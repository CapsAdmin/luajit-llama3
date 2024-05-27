local Tensor = require("tensor")
local ffi = require("ffi")

local function loadTensor(entry)
    print("loading tensor of size " .. tostring(entry.size) .. " and type " .. entry.type_info.name)

    if Tensor[entry.type_info.name] then
        return Tensor[entry.type_info.name](Tensor, entry.size, entry.blob)
    end

    error("NYI tensor type: " .. entry.type_info.name)
end

local function loadTensorArray(size, getTensorEntry, set, get)
    local array = {}

    for i = 1, size do
        array[i-1] = loadTensor(getTensorEntry(i - 1))
    end

    return setmetatable({}, {
        __index = function(s, i) 
            assert(i >= 0)
            assert(i < size)
            assert(array[i])
            print(s, i, array[i], "!??!?!")
            return array[i] 
        end
    })
end

local function Weights(tensors, numberOfLayers) 
    return {
        -- token embedding table
        token_embedding_table = loadTensor(tensors["token_embd.weight"]), -- (vocab_size, dim)
        -- weights for rmsnorms
        rms_att_weight = loadTensorArray(numberOfLayers, function(i)
            return tensors["blk." .. i .. ".attn_norm.weight"]
        end), -- (layer, dim) rmsnorm weights
        -- weights for matmuls
        wq = loadTensorArray(numberOfLayers, function(i)
            return tensors["blk." .. i .. ".attn_q.weight"]
        end), -- (layer, n_heads * head_size)
        wk = loadTensorArray(numberOfLayers, function(i)
            return tensors["blk." .. i .. ".attn_k.weight"]
        end), -- (layer, n_kv_heads, head_size)
        wv = loadTensorArray(numberOfLayers, function(i)
            return tensors["blk." .. i .. ".attn_v.weight"]
        end), -- (layer, n_kv_heads * head_size)
        wo = loadTensorArray(numberOfLayers, function(i)
            return tensors["blk." .. i .. ".attn_output.weight"]
        end), -- (layer, n_heads * head_size, dim)
        rms_ffn_weight = loadTensorArray(numberOfLayers, function(i)
            return tensors["blk." .. i .. ".ffn_norm.weight"]
        end), -- (layer, dim)
        -- weights for ffn
        w1 = loadTensorArray(numberOfLayers, function(i)
            return tensors["blk." .. i .. ".ffn_gate.weight"]
        end), -- w1 -- (layer, hidden_dim, dim)
        w2 = loadTensorArray(numberOfLayers, function(i)
            return tensors["blk." .. i .. ".ffn_down.weight"]
        end), -- w2 -- (layer, dim, hidden_dim)
        w3 = loadTensorArray(numberOfLayers, function(i)
            return tensors["blk." .. i .. ".ffn_up.weight"]
        end), -- w -- (layer, hidden_dim, dim)
        -- public final rmsnorm
        rms_final_weight = loadTensor(tensors["output_norm.weight"]), -- (dim,)
        -- (optional) classifier weights for the logits, on the last layer
        wcls = loadTensor(tensors["output.weight"]), -- (vocab_size, dim)
    }
end

return Weights