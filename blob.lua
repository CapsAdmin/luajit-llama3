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

do
    function Blob:GetF32(index)
        return self.blob[index]
    end

    function Blob:SetF32(index, val)
        self.blob[index] = val
    end

	function Blob:F32(size, blob)
		local blob_ref = blob

		if not blob then
			blob = ffi.cast("float*", ffi.C.malloc(size * 4))
			--ffi.fill(blob, size * 4, 0)
			blob_ref = blob
		end

        local self = setmetatable({}, Blob)
        self.blob = blob
        self.blob_ref = blob_ref -- need to keep this around for gc
        self.size = tonumber(size)
        self.SetFloat = Blob.SetF32
        self.GetFloat = Blob.GetF32

        return self
	end
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

	function Blob:GetQ4_0(index)
		local block_index = rshift(index, 5)
		local block_offset = block_index * type_size
		
		local scale = f16_to_f32(self.blob_f16[block_index * half_type_size])

		local modIndex = band(index, block_size - 1)
		local base_offset = block_offset + band(modIndex, half_block_size - 1)

		local shift_amount = rshift(modIndex, 4) * 4
		local quant = band(rshift(self.blob[2 + base_offset], shift_amount), 0x0F)

		return (quant - 8) * scale
	end

	function Blob:Q4_0(size, blob)
		local blob_ref = blob

		if not blob then
			blob = ffi.cast("uint8_t*", ffi.C.malloc(size))
			ffi.fill(blob, size, 0)
			blob_ref = blob
		end

        local self = setmetatable({}, Blob)
        self.blob = blob
        self.blob_ref = blob_ref -- need to keep this around for gc
        self.size = tonumber(size)
        self.GetFloat = Blob.GetQ4_0
		self.blob_f16 = ffi.cast("uint16_t*", blob)

		return self
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
		local ct = ctype(
			self.size,
			self.GetFloat == Blob.GetF32 and 0 or 1,
			ffi.cast("void *", self.blob)
		)
		return ct
	end

	function Blob:ThreadDeserialize(ptr)
		local data = ffi.cast(ctype_ptr, ptr)

		if data.type == 0 then
			return setmetatable(
				{
					size = data.size,
					blob = ffi.cast("float*", data.blob),
					GetFloat = Blob.GetF32,
					SetFloat = Blob.SetF32,
				},
				Blob
			)
		else
			return setmetatable(
				{
					size = data.size,
					blob = ffi.cast("uint8_t*", data.blob),
					blob_f16 = ffi.cast("uint16_t*", data.blob),
					GetFloat = Blob.GetQ4_0,
				},
				Blob
			)
		end
	end
end

return Blob