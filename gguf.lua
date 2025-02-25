local ffi = require("ffi")
local measure = require("debug.measure")
ffi.cdef[[
	typedef struct FILE FILE;
	size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
	void *malloc( size_t size );
]]
local GGMLType
local GGMLTypeMap = {}

do
	local BYTE = 1
	local FLOAT = 4
	local SHORT = 2
	local INTEGER = 4
	local FLOAT16 = 2 -- Assuming 16-bit half-precision float (2 bytes)
	local list = {
		{name = "F32", type_size = FLOAT, block_size = 1},
		{name = "F16", type_size = FLOAT16, block_size = 1},
		{name = "Q4_0", type_size = FLOAT16 + 16 * BYTE, block_size = 32},
		{name = "Q4_1", type_size = 2 * FLOAT16 + 16 * BYTE, block_size = 32},
		{name = "UNSUPPORTED_Q4_2", type_size = INTEGER, block_size = 1},
		{name = "UNSUPPORTED_Q4_3", type_size = INTEGER, block_size = 1},
		{name = "Q5_0", type_size = INTEGER, block_size = 1},
		{name = "Q5_1", type_size = INTEGER, block_size = 1},
		{name = "Q8_0", type_size = FLOAT16 + 32 * BYTE, block_size = 32},
		{name = "Q8_1", type_size = 32 * BYTE + 2 * FLOAT, block_size = 32},
		{name = "Q2_K", type_size = INTEGER, block_size = 1},
		{name = "Q3_K", type_size = INTEGER, block_size = 1},
		{
			name = "Q4_K",
			type_size = 2 * FLOAT16 + (256 / 16 / 8 * 6) + 256 / 2,
			block_size = 256,
		},
		{
			name = "Q5_K",
			type_size = 2 * FLOAT16 + (256 / 16 / 8 * 6) + 256 / 8 + 256 / 2,
			block_size = 256,
		},
		{
			name = "Q6_K",
			type_size = 256 / 2 + 256 / 4 + 256 / 16 + FLOAT16,
			block_size = 256,
		},
		{name = "Q8_K", type_size = INTEGER, block_size = 1},
		{name = "I8", type_size = BYTE, block_size = 1},
		{name = "I16", type_size = SHORT, block_size = 1},
		{name = "I32", type_size = INTEGER, block_size = 1},
	}

	for i, v in ipairs(list) do
		v.enum = i - 1
		GGMLTypeMap[v.name] = v
	end

	GGMLType = list
end

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
			return reader[assert(types[value_type + 1], "invalid value type " .. value_type)](file)
		end

		reader = {
			function(file) -- "UINT8",
				return read_primitive(file, "uint8_t")
			end,
			function(file) -- "INT8",
				return read_primitive(file, "int8_t")
			end,
			function(file) -- "UINT16",
				return read_primitive(file, "uint16_t")
			end,
			function(file) -- "INT16",
				return read_primitive(file, "int16_t")
			end,
			function(file) -- "UINT32",
				return read_primitive(file, "uint32_t")
			end,
			function(file) -- "INT32",
				return read_primitive(file, "int32_t")
			end,
			function(file) -- "FLOAT32",
				return read_primitive(file, "float")
			end,
			function(file) -- "BOOL",
				return read_primitive(file, "bool")
			end,
			function(file) -- "STRING",
				return file:read(tonumber(reader.UINT64(file)))
			end,
			function(file) -- "ARRAY",
				local value_type = reader.UINT32(file)
				local arr = {}

				for i = 1, tonumber(reader.UINT64(file)) do
					arr[i] = read_value(file, value_type)
				end

				return arr
			end,
			function(file) -- "UINT64",
				return read_primitive(file, "uint64_t")
			end,
			function(file) -- "INT64",
				return read_primitive(file, "int64_t")
			end,
			function(file) -- "FLOAT64",
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
	measure("reading gguf metadata")
	local file = assert(io.open(path, "rb"))
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
		local size = 1

		for i = 1, reader.UINT32(file) do
			local dim = reader.UINT64(file)
			dimensions_map[i] = dim
			size = size * dim
		end

		local type_info = assert(GGMLType[reader.UINT32(file) + 1], "invalid ggml typeid")
		local offset = reader.UINT64(file)
		tensors[i] = {
			name = name,
			size = size,
			type_info = type_info,
			offset = offset,
		}
	end

	measure()
	local alignment = metadata["general.alignment"] or 32
	local padding = alignment - (file:seek() % alignment)
	local pos = file:seek() + padding
	local remaining = file:seek("end")
	file:seek("set", pos)
	local remaining_size = remaining - pos
	local mega_buffer
	measure("reading gguf tensors")
	local mega_buffer = ffi.cast("uint8_t *", ffi.C.malloc(remaining_size))

	if ffi.C.fread(mega_buffer, 1, remaining_size, file) ~= remaining_size then
		file:close()
		error("Failed to read the tensor")
	end

	measure()

	for i, tensor in ipairs(tensors) do
		tensor.blob = mega_buffer + tensor.offset
		tensor.offset = nil
	end

	local tensor_map = {}

	for i, v in ipairs(tensors) do
		tensor_map[v.name] = v
		v.index = i
	end

	return metadata, tensor_map
end

local f16_to_f32
local f16_to_f32_cache
do
	local ldexp = math.ldexp
	local cached_f16_to_f32 = ffi.new("float[65536]")

	for i = 0, 65536 - 1 do
		local sign = 1 - bit.band(bit.rshift(i, 15), 0x1) * 2
		local exponent = bit.band(bit.rshift(i, 10), 0x1F)
		local mantissa = bit.band(i, 0x3FF)
		cached_f16_to_f32[i] = sign * math.ldexp(mantissa + 1024, exponent - 25)
	end

	f16_to_f32 = function(bits) return cached_f16_to_f32[bits] end
	f16_to_f32_cache = cached_f16_to_f32
end

return {load = load_gguf, GGMLType = GGMLType, GGMLTypeMap = GGMLTypeMap, f16_to_f32 = f16_to_f32, f16_to_f32_cache = f16_to_f32_cache}