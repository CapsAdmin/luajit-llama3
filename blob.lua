local Blob = {}
Blob.__index = Blob
local ffi = require("ffi")
ffi.cdef[[
	void *malloc( size_t size );
]]

function Blob:SetFloat(index, val)
	error("NYI")
end

function Blob:GetFloat(index)
	error("NYI", 2)
end

function Blob:Fill(num)
	ffi.fill(self.blob, self.byte_size, num)
	return self
end

function Blob:F32(size, blob)
	blob = blob or ffi.cast("float*", ffi.C.malloc(size * 4))
	blob = ffi.cast("float*", blob)
	return setmetatable(
		{
			type = "F32",
			blob = blob,
			size = tonumber(size),
			byte_size = tonumber(size * 4),
			SetFloat = function(s, index, val)
				blob[index] = val
			end,
			GetFloat = function(s, index)
				return blob[index]
			end,
		},
		Blob
	)
end

do
	local ggf = require("gguf")
	local block_size = ggf.GGMLTypeMap.Q4_0.blockSize
	local half_block_size = block_size / 2
	local type_size = ggf.GGMLTypeMap.Q4_0.typeSize
	local half_type_size = type_size / 2
	local rshift = bit.rshift
	local band = bit.band
	local ldexp = math.ldexp
	local lshift = bit.lshift

	-- f16_to_f32 is not accurate because we don't need to handle nan, inf, etc
	local function f16_to_f32(bits)
		local sign = 1 - band(rshift(bits, 15), 0x1) * 2
		local exponent = band(rshift(bits, 10), 0x1F)
		local mantissa = band(bits, 0x3FF)
		return sign * ldexp(mantissa + 1024, exponent - 25)
	end

	function Blob:Q4_0(size, blob)
		blob = blob or ffi.cast("uint8_t*", ffi.C.malloc(size))
		blob = ffi.cast("uint8_t*", blob)
		local blob_f16 = ffi.cast("uint16_t*", blob)
		
		local byte_size = size * type_size
		assert(byte_size % block_size == 0, "Total size must be a multiple of the block size")
		byte_size = byte_size / block_size

		return setmetatable(
			{
				type = "Q4_0",
				blob = blob,
				size = tonumber(size),
				byte_size = tonumber(byte_size),
				blob_f16 = blob_f16,
				GetFloat = function(s, index)
					local block_index = rshift(index, 5)
					local block_offset = block_index * type_size
					local scale = f16_to_f32(blob_f16[block_index * half_type_size])
					local modIndex = band(index, block_size - 1)
					local base_offset = block_offset + band(modIndex, half_block_size - 1)
					local shift_amount = rshift(modIndex, 4) * 4
					local quant = band(rshift(blob[2 + base_offset], shift_amount), 0x0F)
					return (quant - 8) * scale
				end,
			},
			Blob
		)
	end
end

do
	local ctype = ffi.typeof([[
		struct {
			int size;
			int type;
			void *blob; 
		}
	]])
	local ctype_ptr = ffi.typeof("$*", ctype)
	local ctype_box = ffi.typeof("$[1]", ctype)
	local type_map = {
		F32 = 0,
		Q4_0 = 1,
	}

	-- double lookup
	for i, str in pairs(type_map) do
		type_map[str] = i
	end

	function Blob:ThreadSerialize()
		return ctype(self.size, type_map[self.type], ffi.cast("void *", self.blob))
	end

	function Blob:ThreadDeserialize(ptr)
		local data = ffi.cast(ctype_ptr, ptr)

		if not type_map[data.type] then error("unknown type " .. data.type) end

		return Blob[type_map[data.type]](Blob, data.size, data.blob)
	end
end

return Blob