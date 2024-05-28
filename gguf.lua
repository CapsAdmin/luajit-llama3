local ffi = require("ffi")
ffi.cdef[[
	typedef struct FILE FILE;
	size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
	void *malloc( size_t size );
]]
local GGMLType, GGMLTypeMap = (
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

		local map = {}
		for i, v in ipairs(GGMLType) do
			v.enum = i - 1
			map[v.name] = v
		end


		return GGMLType, map
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
	_G.timer("reading gguf metadata")
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

	_G.timer()
	
	local alignment = metadata["general.alignment"] or 32
	local padding = alignment - (file:seek() % alignment)
	local pos = file:seek() + padding
	local remaining = file:seek("end")
	file:seek("set", pos)
	local remaining_size = remaining-pos
	
	local mega_buffer
	_G.timer("reading gguf tensors")
	local mega_buffer = ffi.cast("uint8_t *", ffi.C.malloc(remaining_size))
	if ffi.C.fread(mega_buffer, 1, remaining_size, file) ~= remaining_size then
		file:close()
		error("Failed to read the tensor")
	end
	_G.timer()

	for i, tensor in ipairs(tensors) do
		tensor.blob = mega_buffer + tensor.offset

		-- random sanity check
		if tensor.name == "blk.0.attn_norm.weight" then
			assert(ffi.cast("float*", tensor.blob)[585] == 0.5625)
		end
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

return {load_gguf = load_gguf,GGMLType = GGMLType, GGMLTypeMap = GGMLTypeMap }
