local TEST = [[
John's quick to resp_ond, isn't he? I've seen 123 birds, mostly sparrows, bluebirds, and crows. They're not "typical" birds; these'll fly up to 300 miles in a day! However, I'm not too keen on studying them further. Haven't you heard? It's strange, really.

-- "Yes," she said, "they've adapted well to urban life."

No doubt, innovations in city planning—especially those focused on sustainability—will need to account for such wildlife. Urban environments, you know? They're teeming with life; not just birds, but also small mammals, insects, and plants. Oh, and remember, they'll need 'water sources' too. 

Can't forget about the $%&* symbols in texts, right? Or punctuation!!! These elements can add "flavor" to any written piece.

So, what'll it be? More studies or just admiring from afar?	
]]
local ffi = require("ffi")
-- Define sizes manually
local GGMLType = (
	function()
		local BYTE = 1
		local FLOAT = 4
		local SHORT = 2
		local INTEGER = 4
		local FLOAT16 = 2 -- Assuming 16-bit half-precision float (2 bytes)
		local GGMLType = {
			{name = "F32", typeSize = FLOAT, blockSize = 1},
			{name = "F16", typeSize = FLOAT16, blockSize = 1},
			{name = "Q4_0", typeSize = FLOAT16 + 16 * BYTE, blockSize = 32},
			{name = "Q4_1", typeSize = 2 * FLOAT16 + 16 * BYTE, blockSize = 32},
			{name = "UNSUPPORTED_Q4_2", typeSize = INTEGER, blockSize = 1},
			{name = "UNSUPPORTED_Q4_3", typeSize = INTEGER, blockSize = 1},
			{name = "Q5_0", typeSize = INTEGER, blockSize = 1},
			{name = "Q5_1", typeSize = INTEGER, blockSize = 1},
			{name = "Q8_0", typeSize = FLOAT16 + 32 * BYTE, blockSize = 32},
			{name = "Q8_1", typeSize = 32 * BYTE + 2 * FLOAT, blockSize = 32},
			{name = "Q2_K", typeSize = INTEGER, blockSize = 1},
			{name = "Q3_K", typeSize = INTEGER, blockSize = 1},
			{
				name = "Q4_K",
				typeSize = 2 * FLOAT16 + (256 / 16 / 8 * 6) + 256 / 2,
				blockSize = 256,
			},
			{
				name = "Q5_K",
				typeSize = 2 * FLOAT16 + (256 / 16 / 8 * 6) + 256 / 8 + 256 / 2,
				blockSize = 256,
			},
			{
				name = "Q6_K",
				typeSize = 256 / 2 + 256 / 4 + 256 / 16 + FLOAT16,
				blockSize = 256,
			},
			{name = "Q8_K", typeSize = INTEGER, blockSize = 1},
			{name = "I8", typeSize = BYTE, blockSize = 1},
			{name = "I16", typeSize = SHORT, blockSize = 1},
			{name = "I32", typeSize = INTEGER, blockSize = 1},
		}

		for i, v in ipairs(GGMLType) do
			v.enum = i - 1
		end

		return GGMLType
	end
)()
local read_primitive = (
	function()
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
		return function(file, cdecl)
			local info = cached_info(cdecl)
			local str = file:read(info.size)
			local ptr = ffi.cast(void_ptr, str)
			return ffi.cast(info.typ_ptr, ptr)[0]
		end
	end
)()
local reader, read_value = (
	function()
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

		return reader, read_value
	end
)()

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

	local function get_tensor_size(dimensions_map)
		local t = 1

		for i, v in ipairs(dimensions_map) do
			t = t * v
		end

		return t
	end

	local function get_tensor_byte_size(type_info, dimensions_map)
		local t = get_tensor_size(dimensions_map)
		t = t * type_info.typeSize
		assert(t % type_info.blockSize == 0, "Total size must be a multiple of the block size")
		t = t / type_info.blockSize
		return t
	end

	local tensors = {}

	for i = 1, tonumber(tensor_count) do
		local name = reader.STRING(file)
		local dimensions_map = {}

		for i = 1, reader.UINT32(file) do
			dimensions_map[i] = reader.UINT64(file)
		end

		local type_info = assert(GGMLType[reader.UINT32(file) + 1], "invalid ggml typeid")
		local offset = reader.UINT64(file)
		tensors[i] = {
			name = name,
			byte_size = get_tensor_byte_size(type_info, dimensions_map),
			size = get_tensor_size(dimensions_map),
			ggml_typeid = ggml_typeid,
			type_info = type_info,
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

	for i, v in ipairs(tensors) do
		tensor_map[v.name] = v
		v.index = i
	end

	return {
		metadata = metadata,
		tensors = tensor_map,
	}
end

local function Tokenizer(gguf)
	local function Vocabulary(gguf)
		assert(gguf.metadata["tokenizer.ggml.model"] == "gpt2")
		local tokens = gguf.metadata["tokenizer.ggml.tokens"]
		local tokens_map = {}

		for i, v in ipairs(tokens) do
			tokens_map[v] = i
		end

		return {
			get = function(i)
				return tokens[i]
			end,
			getIndex = function(str)
				return tokens_map[str]
			end,
			size = function()
				return #tokens
			end,
			token_map = tokens_map,
			tokens = tokens,
		}
	end

	local vocabulary = Vocabulary(gguf)
	local merges = {} -- passed to tokenizer
	do -- merge lines
		for k, v in ipairs(gguf.metadata["tokenizer.ggml.merges"]) do
			local l, r = v:match("(%S+) (%S+)")
			table.insert(merges, {vocabulary.getIndex(l), vocabulary.getIndex(r)})
		end
	end

	local all_tokens = vocabulary.size()
	local base_tokens = 128000
	local reservedSpecialTokens = all_tokens - base_tokens
	local specialTokensList = {}

	for i = base_tokens, all_tokens do
		table.insert(specialTokensList, vocabulary.get(i))
	end

	local specialTokens = {}

	for i = 1, #specialTokensList do
		specialTokens[specialTokensList[i]] = base_tokens + i
	end

	local LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
	-- actual tokenizer here
	local merges2 = {}

	for _, pair in ipairs(merges) do
		merges2[pair[1] .. "|" .. pair[2]] = vocabulary.getIndex(vocabulary.get(pair[1]) .. vocabulary.get(pair[2]))
	end

	local function utf8_uint32(char, offset)
		if char == "" then return -1 end

		offset = offset or 1
		local byte = char:byte(offset)

		if byte and byte >= 128 then
			if byte >= 240 then
				if #char < 4 then return -1 end

				byte = (byte % 8) * 262144
				byte = byte + (char:byte(offset + 1) % 64) * 4096
				byte = byte + (char:byte(offset + 2) % 64) * 64
				byte = byte + (char:byte(offset + 3) % 64)
			elseif byte >= 224 then
				if #char < 3 then return -1 end

				byte = (byte % 16) * 4096
				byte = byte + (char:byte(offset + 1) % 64) * 64
				byte = byte + (char:byte(offset + 2) % 64)
			elseif byte >= 192 then
				if #char < 2 then return -1 end

				byte = (byte % 32) * 64
				byte = byte + (char:byte(offset + 1) % 64)
			else
				byte = -1
			end
		end

		return byte
	end

	function bytesToUnicode()
		local function table_contains(tbl, item)
			for _, value in pairs(tbl) do
				if value == item then return true end
			end

			return false
		end

		local bs = {}

		for i = utf8_uint32("!"), utf8_uint32("~") do
			table.insert(bs, i)
		end

		for i = utf8_uint32("¡"), utf8_uint32("¬") do
			table.insert(bs, i)
		end

		for i = utf8_uint32("®"), utf8_uint32("ÿ") do
			table.insert(bs, i)
		end

		local cs = {}

		for _, v in ipairs(bs) do
			table.insert(cs, v)
		end

		local n = 0

		for b = 0, 255 do
			if not table_contains(bs, b) then
				table.insert(bs, b)
				table.insert(cs, 256 + n)
				n = n + 1
			end
		end

		local result = {}

		for i = 1, #bs do
			result[bs[i]] = cs[i]
		end

		return result
	end

	local BYTESMAP = bytesToUnicode()

	local function encode(str)
		local function encode(str, special)
			if not special then encode_ordinary(str) end

			for k, v in pairs(special) do
				assert(specialTokens[k])
			end

			local specialPattern = {}

			for k, v in pairs(special) do
				table.insert(specialPattern, "\"" .. specialPattern .. "\"")
			end

			specialPattern = "(" .. table.concat(specialPattern, "|") .. ")"
		end

		local function get_stats(ids)
			local map = {}

			for i = 1, #ids - 1 do
				local key = ids[i + 0] .. "|" .. ids[i + 1]
				map[key] = (map[key] or 0) + 1
			end

			return map
		end

		local function merge(ids, pair, idx)
			local newIds = {}
			local i = 1

			while i <= #ids do
				if pair == (ids[i] .. "|" .. (ids[i + 1] or "-")) then
					table.insert(newIds, idx)
					i = i + 2
				else
					table.insert(newIds, ids[i])
					i = i + 1
				end
			end

			return newIds
		end

		local function utf8_encode(cp)
			if cp < 0x80 then
				return string.char(cp)
			elseif cp < 0x800 then
				return string.char(0xC0 + math.floor(cp / 0x40), 0x80 + (cp % 0x40))
			elseif cp < 0x10000 then
				return string.char(
					0xE0 + math.floor(cp / 0x1000),
					0x80 + (math.floor(cp / 0x40) % 0x40),
					0x80 + (cp % 0x40)
				)
			elseif cp < 0x200000 then
				return string.char(
					0xF0 + math.floor(cp / 0x40000),
					0x80 + (math.floor(cp / 0x1000) % 0x40),
					0x80 + (math.floor(cp / 0x40) % 0x40),
					0x80 + (cp % 0x40)
				)
			end
		end

		local function MERGE(str)
			local ids = {}

			for i = 1, #str do
				local b = utf8_uint32(str, i)

				if b ~= -1 then -- TODO, get length of utf8 string
					local index = vocabulary.getIndex(utf8_encode(b))
					table.insert(ids, index)
				end
			end

			while #ids > 2 do
				local stats = get_stats(ids)
				local minPair
				local minIndex = math.huge

				for pair, index in pairs(stats) do
					local mergeIndex = merges2[pair] or math.huge

					if mergeIndex < minIndex then
						minIndex = mergeIndex
						minPair = pair
					end
				end

				if not merges2[minPair] then
					break -- Nothing else can be merged anymore
				end

				local idx = merges2[minPair]
				ids = merge(ids, minPair, idx)
			end

			return ids
		end

		local new = {}

		for i = 1, #str do
			local test = str:byte(i)
			local int = BYTESMAP[test]
			table.insert(new, utf8_encode(int))
		end

		new = table.concat(new)

		local tokens = {}

		for i,v in ipairs(MERGE(new)) do
			tokens[i] = v
		end

		return tokens
	end

	do
		local tks = encode("hello world")
		assert(#tks == 2)
		assert(tks[1] == 15340)
		assert(tks[2] == 1918)
	end

	do
		local tks =encode[[John's quick to respond, isn't he? I've seen 123 birds, mostly sparrows, bluebirds, and crows. They're not "typical" birds; these'll fly up to 300 miles in a day! However, I'm not too keen on studying them further. Haven't you heard? It's strange, really.

-- "Yes," she said, "they've adapted well to urban life."

No doubt, innovations in city planning—especially those focused on sustainability—will need to account for such wildlife. Urban environments, you know? They're teeming with life; not just birds, but also small mammals, insects, and plants. Oh, and remember, they'll need 'water sources' too. 

Can't forget about the $%&* symbols in texts, right? Or punctuation!!! These elements can add "flavor" to any written piece.

So, what'll it be? More studies or just admiring from afar?]]
	
		for k,v in pairs(tks) do
			print(k,v)
		end
	end
	return {
		vocabulary = vocabulary,
		merges = merges,
		specialTokens = specialTokens,
		tokens = tokens,
	}
end

local gguf = load_gguf("/home/caps/projects/llama3.java/Meta-Llama-3-8B-Instruct-Q4_0.gguf")
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
local vocabularySize = tokenizer.vocabulary.size()
local contextLength = context_length
local sharedWeights = false
local rmsNormEps = gguf.metadata["llama.attention.layer_norm_rms_epsilon"] or 1e-5
local ropeTheta = gguf.metadata["llama.rope.freq_base"] or 10000
local headSize = dim / numberOfHeads

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

local ropeFreqsReal, ropeFreqsImag = rope_freq(contextLength, headSize, ropeTheta)

local function Q4_0FloatTensor(size, blob)
	local self = {}
	self.size = size
	self.type = GGMLType.Q4_0

	function self:SetFloat(index, value)
		error("NYI")
	end

	function self:GetFloatVector(index, value)
		error("NYI")
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
			quant = bit.band(
				bit.rshift(blob[block_offset + FLOAT16 + modIndex - self.type.blockSize / 2], 4),
				0x0F
			)
		end

		quant = quant - 8
		return quant * scale
	end

	function self:Dot(thisOffset, that, thatOffset, size)
		local result = 0

		for j = 0, size - 1 do
			result = result + self:GetFloat(thisOffset + j) * that:GetFloat(thatOffset + j)
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

local function F32FloatTensor(size, blob)
	local self = {}
	self.size = size
	self.type = GGMLType.Q4_0

	function self:SetFloat(index, value)
		blob[index] = value
	end

	function self:GetFloatVector(index, value)
		error("NYI")
	end

	function self:GetFloat(index)
		return blob[index]
	end

	function self.ScalarDot(self, thisOffset, that, thatOffset, size)
		local result = 0

		for j = 0, size - 1 do
			result = result + self:GetFloat(thisOffset + j) * that:GetFloat(thatOffset + j)
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
		return self:Reduce(thisOffset, size, 0, function(r, f)
			return r + f
		end)
	end

	function self:Max(thisOffset, size)
		return self:Reduce(thisOffset, size, 0, function(r, f)
			return math.max(r, f)
		end)
	end

	function self:CopyT(thisOffset, that, thatOffset, size)
		return self:Reduce(thisOffset, size, 0, function(r, f) end)
	end

	return self
end

local function loadQuantized(entry)
	if entry.type_info.name == "Q4_0" then
		return Q4_0FloatTensor(entry.size, entry.blob)
	else
		error("unsupported quant format " .. entry.type_info.name)
	end
end

local function toFloatBuffer(entry)
	if entry.type_info.name == "F32" then
		return entry.blob -- as float[]
	else
		error("unsupported quant format " .. entry.type_info.name)
	end
end

local function loadArrayOfFloatBuffer(size, getTensorEntry)
	local array = {}

	for i = 1, size do
		array[i] = toFloatBuffer(getTensorEntry(i - 1))
	end

	return array
end

local function loadArrayOfQuantized(size, getTensorEntry)
	local array = {}

	for i = 1, size do
		array[i] = loadQuantized(getTensorEntry(i - 1))
	end

	return array
end

local function llama_weights(weights) end

for k, v in pairs(gguf.tensors) do
	print(k, v)
end

llama_weights(
	{
		-- token embedding table
		token_embedding_table = loadQuantized(gguf.tensors["token_embd.weight"]), -- (vocab_size, dim)
		-- weights for rmsnorms
		rms_att_weight = loadArrayOfFloatBuffer(numberOfLayers, function(i)
			return gguf.tensors["blk." .. i .. ".attn_norm.weight"]
		end), -- (layer, dim) rmsnorm weights
		-- weights for matmuls
		wq = loadArrayOfQuantized(numberOfLayers, function(i)
			return gguf.tensors["blk." .. i .. ".attn_q.weight"]
		end), -- (layer, n_heads * head_size)
		wk = loadArrayOfQuantized(numberOfLayers, function(i)
			return gguf.tensors["blk." .. i .. ".attn_k.weight"]
		end), -- (layer, n_kv_heads, head_size)
		wv = loadArrayOfQuantized(numberOfLayers, function(i)
			return gguf.tensors["blk." .. i .. ".attn_v.weight"]
		end), -- (layer, n_kv_heads * head_size)
		wo = loadArrayOfQuantized(numberOfLayers, function(i)
			return gguf.tensors["blk." .. i .. ".attn_output.weight"]
		end), -- (layer, n_heads * head_size, dim)
		rms_ffn_weight = loadArrayOfFloatBuffer(numberOfLayers, function(i)
			return gguf.tensors["blk." .. i .. ".ffn_norm.weight"]
		end), -- (layer, dim)
		-- weights for ffn
		w1 = loadArrayOfQuantized(numberOfLayers, function(i)
			return gguf.tensors["blk." .. i .. ".ffn_gate.weight"]
		end), -- w1 -- (layer, hidden_dim, dim)
		w2 = loadArrayOfQuantized(numberOfLayers, function(i)
			return gguf.tensors["blk." .. i .. ".ffn_down.weight"]
		end), -- w2 -- (layer, dim, hidden_dim)
		w3 = loadArrayOfQuantized(numberOfLayers, function(i)
			return gguf.tensors["blk." .. i .. ".ffn_up.weight"]
		end), -- w -- (layer, hidden_dim, dim)
		-- public final rmsnorm
		rms_final_weight = toFloatBuffer(gguf.tensors["output_norm.weight"]), -- (dim,)
		-- freq_cis for RoPE relatively positional embeddings
		freq_cis_real = ropeFreqsReal, -- (seq_len, head_size/2)
		freq_cis_imag = ropeFreqsImag, -- (seq_len, head_size/2)
		-- (optional) classifier weights for the logits, on the last layer
		wcls = loadQuantized(gguf.tensors["output.weight"]), -- (vocab_size, dim)
	}
)