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

function Blob:F32(size, blob)
	blob = blob or ffi.cast("float*", ffi.C.malloc(size * 4))
	return setmetatable(
		{
			type = "F32",
			blob = ffi.cast("float*", blob),
			size = tonumber(size),
			SetFloat = function(s, index, val)
				s.blob[index] = val
			end,
			GetFloat = function(s, index)
				return s.blob[index]
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
		return setmetatable(
			{
				type = "Q4_0",
				blob = ffi.cast("uint8_t*", blob),
				size = tonumber(size),
				blob_f16 = ffi.cast("uint16_t*", blob),
				GetFloat = function(s, index)
					local block_index = rshift(index, 5)
					local block_offset = block_index * type_size
					local scale = f16_to_f32(s.blob_f16[block_index * half_type_size])
					local modIndex = band(index, block_size - 1)
					local base_offset = block_offset + band(modIndex, half_block_size - 1)
					local shift_amount = rshift(modIndex, 4) * 4
					local quant = band(rshift(s.blob[2 + base_offset], shift_amount), 0x0F)
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

	function Blob:ThreadSerialize()
		local ct = ctype(self.size, self.type == "F32" and 0 or 1, ffi.cast("void *", self.blob))
		return ct
	end

	function Blob:ThreadDeserialize(ptr)
		local data = ffi.cast(ctype_ptr, ptr)

		if data.type == 0 then
			return Blob:F32(data.size, data.blob)
		else
			return Blob:Q4_0(data.size, data.blob)
		end
	end
end

return Blob