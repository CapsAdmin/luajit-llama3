local ffi = require("ffi")
-- Define sizes manually
local BYTE = 1
local FLOAT = 4
local SHORT = 2
local INTEGER = 4
local FLOAT16 = 2 -- Assuming 16-bit half-precision float (2 bytes)
local GGMLType = {
	F32 = {enum = 0, typeSize = FLOAT, blockSize = 1},
	F16 = {enum = 1, typeSize = FLOAT16, blockSize = 1},
	Q4_0 = {enum = 2, typeSize = FLOAT16 + 16 * BYTE, blockSize = 32},
	Q4_1 = {enum = 3, typeSize = 2 * FLOAT16 + 16 * BYTE, blockSize = 32},
	UNSUPPORTED_Q4_2 = {enum = 4, typeSize = INTEGER, blockSize = 1}, -- support removed, handling similarly
	UNSUPPORTED_Q4_3 = {enum = 5, typeSize = INTEGER, blockSize = 1}, -- support removed, handling similarly
	Q5_0 = {enum = 6, typeSize = INTEGER, blockSize = 1},
	Q5_1 = {enum = 7, typeSize = INTEGER, blockSize = 1},
	Q8_0 = {enum = 8, typeSize = FLOAT16 + 32 * BYTE, blockSize = 32},
	Q8_1 = {enum = 9, typeSize = 32 * BYTE + 2 * FLOAT, blockSize = 32},
	Q2_K = {enum = 10, typeSize = INTEGER, blockSize = 1},
	Q3_K = {enum = 11, typeSize = INTEGER, blockSize = 1},
	Q4_K = {
		enum = 12,
		typeSize = 2 * FLOAT16 + (256 / 16 / 8 * 6) + 256 / 2,
		blockSize = 256,
	},
	Q5_K = {
		enum = 13,
		typeSize = 2 * FLOAT16 + (256 / 16 / 8 * 6) + 256 / 8 + 256 / 2,
		blockSize = 256,
	},
	Q6_K = {enum = 14, typeSize = 256 / 2 + 256 / 4 + 256 / 16 + FLOAT16, blockSize = 256},
	Q8_K = {enum = 15, typeSize = INTEGER, blockSize = 1},
	I8 = {enum = 16, typeSize = BYTE, blockSize = 1},
	I16 = {enum = 17, typeSize = SHORT, blockSize = 1},
	I32 = {enum = 18, typeSize = INTEGER, blockSize = 1},
}

local function ggml_type_from_enum(i)
	for k, v in pairs(GGMLType) do
		if v.enum == i then return v, k end
	end
end

local function get_tensor_size(type, dimensions_map)
	local ggml_type = ggml_type_from_enum(type)

	local t = 1
	
	for i, v in ipairs(dimensions_map) do
		t = t * v
	end

	return t
end

local function get_tensor_byte_size(type, dimensions_map)
	local t = get_tensor_size(type, dimensions_map)
	local ggml_type = ggml_type_from_enum(type)

	t = t * ggml_type.typeSize
	assert(t % ggml_type.blockSize == 0, "Total size must be a multiple of the block size")
	t = t / ggml_type.blockSize

	return t
end

local function rope_freq(context_length, head_size, theta)
	assert(head_size % 2 == 0)
	local cr = {}
	local ci = {}
	local n = 1

	for pos = 0, context_length - 1 do
		for i = 0, head_size - 1, 2 do
			local freq = 1.0 / (theta ^ (i / head_size))
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



local cache = {}
local cached_info = function(cdecl)
	if cache[cdecl] then return cache[cdecl] end

	local typ = ffi.typeof(cdecl)
	local typ_ptr = ffi.typeof("$*", typ)
	local size = ffi.sizeof(cdecl)
	local val = {
		typ = typ,
		typ_ptr = typ_ptr,
		size = size,
	}
	cache[cdecl] = val
	return val
end
local void_ptr = ffi.typeof("void *")

local function read_primitive(file, cdecl)
	local info = cached_info(cdecl)
	local str = file:read(info.size)
	local ptr = ffi.cast(void_ptr, str)
	return ffi.cast(info.typ_ptr, ptr)[0]
end

local types = {
	"UINT8",
	"INT8",
	"UINT16",
	"INT16",
	"UINT32",
	"INT32",
	"FLOAT32",
	"BOOL",
	"STRING",
	"ARRAY",
	"UINT64",
	"INT64",
	"FLOAT64",
}
local reader

local function read_value(file, value_type)
	return reader[assert(types[value_type + 1], "invalid value type")](file)
end

reader = {
	function(file)
		return read_primitive(file, "uint8_t")
	end,
	function(file)
		return read_primitive(file, "int8_t")
	end,
	function(file)
		return read_primitive(file, "uint16_t")
	end,
	function(file)
		return read_primitive(file, "int16_t")
	end,
	function(file)
		return read_primitive(file, "uint32_t")
	end,
	function(file)
		return read_primitive(file, "int32_t")
	end,
	function(file)
		return read_primitive(file, "float")
	end,
	function(file)
		return read_primitive(file, "bool")
	end,
	function(file)
		return file:read(tonumber(reader.UINT64(file)))
	end,
	function(file)
		local value_type = reader.UINT32(file)
		local arr = {}

		for i = 1, tonumber(reader.UINT64(file)) do
			arr[i] = read_value(file, value_type)
		end

		return arr
	end,
	function(file)
		return read_primitive(file, "uint64_t")
	end,
	function(file)
		return read_primitive(file, "int64_t")
	end,
	function(file)
		return read_primitive(file, "double")
	end,
}

for i, v in ipairs(reader) do
	reader[types[i]] = v
	reader[i] = nil
end

local function load_gguf(path)
	local file = assert(io.open(path, "r"))
	assert(file:read(4) == "GGUF", "not a gguf file")

	do
		local version = reader.UINT32(file)

		if version ~= 2 and version ~= 3 then
			error("unsupported gguf version " .. version)
		end
	end

	local tensor_count = reader.UINT64(file)
	local metadata = {}

	for i = 1, tonumber(reader.UINT64(file)) do
		local key = reader.STRING(file)
		local val = read_value(file, reader.UINT32(file))
		metadata[key] = val
	end

	local tensors = {}

	for i = 1, tonumber(tensor_count) do
		local name = reader.STRING(file)
		local dimensions_map = {}

		for i = 1, reader.UINT32(file) do
			dimensions_map[i] = reader.UINT64(file)
		end

		local ggml_type = reader.UINT32(file)
		local offset = reader.UINT64(file)

		tensors[i] = {
			name = name,
			byte_size = get_tensor_byte_size(ggml_type, dimensions_map),
			size = get_tensor_size(ggml_type, dimensions_map),
			ggml_type = ggml_type,
			offset = offset,
		}
	end

	local alignment = metadata["general.alignment"] or 32
	local padding = alignment - (file:seek() % alignment)
	file:seek("set", file:seek() + padding)

	local tensor_data_offset = file:seek()
	local tensor_data = file:read()

	for i, tensor in ipairs(tensors) do
		file:seek("set", tonumber(tensor.offset))
		local tensor_blob = file:read(tonumber(tensor.byte_size))

		tensor.blob = tensor_blob
	end

	local tensor_map = {}
	for i,v in ipairs(tensors) do
		tensor_map[v.name] = v
		v.index = i
	end

	return {
		metadata = metadata,
		tensors = tensor_map,
	}
end

local function Tokenizer(gguf)
	assert(gguf.metadata["tokenizer.ggml.model"] == "gpt2")
	local tokens = gguf.metadata["tokenizer.ggml.tokens"]
	local merge_lines = gguf.metadata["tokenizer.ggml.merges"]
	local tokens_map = {}

	for k, v in pairs(tokens) do
		tokens_map[v] = k - 1
	end

	local merges = {}

	for k, v in ipairs(merge_lines) do
		local l, r = v:match("(%S+) (%S+)")
		table.insert(merges, {l, r})
	end

	local all_tokens = #tokens
	local base_tokens = 128000
	local reserved = all_tokens - base_tokens
	local special = {}
	local special_tokens_map = {}

	for i = base_tokens, all_tokens - 1 do
		table.insert(special, tokens[i])
		assert(tokens_map[tokens[i]])
		special_tokens_map[tokens_map[tokens[i]]] = base_tokens + #special - 1
	end

	local tk_merges = {}

	for i, v in ipairs(merges) do
		local merge_index = tokens_map[v[1]] + tokens_map[v[2]]
		tk_merges[i] = merge_index
	end

	return {
		tokens = tokens,
	}
end


local gguf = load_gguf(
	"/home/caps/.ollama/models/blobs/sha256:00e1317cbf74d901080d7100f57580ba8dd8de57203072dc6f668324ba545f29"
)

local tokenizer = Tokenizer(gguf)



local context_length = 10000
local model_context_length = gguf.metadata["llama.context_length"]
local dim = gguf.metadata["llama.embedding_length"]
local hiddenDim = gguf.metadata["llama.feed_forward_length"]
local numberOfLayers = gguf.metadata["llama.block_count"]
local numberOfHeads = gguf.metadata["llama.attention.head_count"]
local numberOfKeyValueHeads = gguf.metadata["llama.attention.head_count_kv"] and
gguf.metadata["llama.attention.head_count_kv"] or
gguf.metadata["llama.attention.head_count"]
local vocabularySize = #tokenizer.tokens
local contextLength = context_length
local sharedWeights = false
local rmsNormEps = gguf.metadata["llama.attention.layer_norm_rms_epsilon"] or 1e-5
local ropeTheta = gguf.metadata["llama.rope.freq_base"] or 10000
local headSize = dim / numberOfHeads

local ropeFreqsReal, ropeFreqsImag = rope_freq(contextLength, headSize, ropeTheta)

local function Q4_0FloatTensor(size, blob)
	local self = {}
	self.size = size
	self.type = GGMLType.Q4_0

	function self:SetFloat(index, value)
		error"NYI"
	end	

	function self:GetFloatVector(index, value)
		error"NYI"
	end

	function self:GetFloat(index)
		local block_index = index / self.type.blockSize
		local block_offset = block_index * self.type.typeSize
		local scale = blob[index] -- read uint16_t
		local quant 
		local modIndex = index % self.type.blockSize
		if modIndex < self.type.blockSize / 2 then
			quant = bit.band(blob[block_offset + FLOAT16 + modIndex], 0x0F)
		else
			quant = bit.band(bit.rshift(blob[block_offset + FLOAT16 + modIndex - self.type.blockSize / 2], 4), 0x0F)
		end
		quant = quant - 8

		return quant * scale
	end

	function self:Dot(thisOffset, that, thatOffset, size)
		local result = 0;
		for j = 0, size - 1 do
			result = result + self:GetFloat(thisOffset + j) * that:GetFloat(thatOffset + j);
		end
		return result
	end

	function self:MatMul(that, out, dim0, dim1)
		for i = 0, dim0 - 1 do
			out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
		end		
	end
	
	return self
end

local function FloatTensor(size, blob)
	local self = {}
	self.size = size
	self.type = GGMLType.Q4_0

	function self:SetFloat(index, value)
		error"NYI"
	end	

	function self:GetFloatVector(index, value)
		error"NYI"
	end

	function self:GetFloat(index)
		local block_index = index / self.type.blockSize
		local block_offset = block_index * self.type.typeSize
		local scale = blob[index] -- read uint16_t
		local quant 
		local modIndex = index % self.type.blockSize
		if modIndex < self.type.blockSize / 2 then
			quant = bit.band(blob[block_offset + FLOAT16 + modIndex], 0x0F)
		else
			quant = bit.band(bit.rshift(blob[block_offset + FLOAT16 + modIndex - self.type.blockSize / 2], 4), 0x0F)
		end
		quant = quant - 8

		return quant * scale
	end

	function self.ScalarDot(self, thisOffset, that, thatOffset, size)
		local result = 0;
		for j = 0, size - 1 do
			result = result + self:GetFloat(thisOffset + j) * that:GetFloat(thatOffset + j);
		end
		return result
	end

	function self:Dot(thisOffset, that, thatOffset, size)
		return self.ScalarDot(self, thisOffset, that, thatOffset, size)
	end

	function self:MatMul(that, out, dim0, dim1)
		for i = 0, dim0 - 1 do
			out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
		end		
	end

	function self:Reduce(thisOffset, size, seed, reduce_callback)
		local result = seed
		for i = 0, size - 1 do
			result = reduce_callback(result, self:GetFloat(thisOffset + i))
		end
		return result
	end

	function self:Sum(thisOffset, size)
		return self:Reduce(thisOffset, size, 0, function(r, f) return r + f end)
	end

	function self:Max(thisOffset, size)
		return self:Reduce(thisOffset, size, 0, function(r, f) return math.max(r, f) end)
	end

	function self:CopyT(thisOffset, that, thatOffset, size)
		return self:Reduce(thisOffset, size, 0, function(r, f)  end)
	end
	
	return self
end


local function loadQuantized(entry)
	local info, name = ggml_type_from_enum(entry.ggml_type)

	if name == "Q4_0" then
		return Q4_0FloatTensor(entry.size, entry.blob)
	else
		error("unsupported quant format " .. name)
	end
end

loadQuantized(gguf.tensors["token_embd.weight"])