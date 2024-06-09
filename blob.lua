local Blob = {}
Blob.__index = Blob
local ffi = require("ffi")
ffi.cdef[[
	void *malloc( size_t size );
	void *memcpy(void *dest, const void *src, size_t n);
]]

function Blob:SetFloat(index, val)
	error("NYI")
end

function Blob:GetFloat(index)
	error("NYI", 2)
end

function Blob:CopyTo(thisOffset, that, thatOffset, size)
	if self.type == "F32" and that.type == "F32" then
		ffi.C.memcpy(that.blob + thatOffset, self.blob + thisOffset, size * self.byte_stride)
	else
		for i = thatOffset, thatOffset + size - 1 do
			that:SetFloat(i, self:GetFloat(i - thatOffset + thisOffset))
		end
	end
end

function Blob:Fill(thisOffset, size, identity)
	if self.type == "F32" then
		if identity == 0 then
			ffi.fill(self.blob + thisOffset, size * self.byte_stride, 0)
		else
			for i = thisOffset, thisOffset + size - 1 do
				self:SetFloat(i, identity)
			end
		end
	else
		error("NYI", 2)
	end	
	return self
end

function Blob:F32(size, blob)
	local stride = 4
	blob = ffi.cast("float*", blob or ffi.cast("float*", ffi.C.malloc(size * stride)))
	return setmetatable(
		{
			type = "F32",
			blob = blob,
			size = tonumber(size),
			byte_size = tonumber(size * stride),
			byte_stride = stride,
			SetFloat = function(s, index, val)
				blob[index] = val
			end,
			GetFloat = function(s, index)
				return blob[index]
			end,
		},
		Blob
	)-- :Fill(0, size, 0)
end

function Blob:F64(size, blob)
	local stride = ffi.sizeof("double")
	blob = ffi.cast("double*", blob or ffi.cast("double*", ffi.C.malloc(size * stride)))
	return setmetatable(
		{
			type = "F64",
			blob = blob,
			size = tonumber(size),
			byte_size = tonumber(size * stride),
			byte_stride = stride,
			SetFloat = function(s, index, val)
				blob[index] = val
			end,
			GetFloat = function(s, index)
				return blob[index]
			end,
		},
		Blob
	)-- :Fill(0, size, 0)
end

do
	local ggf = require("gguf")
	local block_size = ggf.GGMLTypeMap.Q4_0.block_size
	local half_block_size = block_size / 2
	local type_size = ggf.GGMLTypeMap.Q4_0.type_size
	local half_type_size = type_size / 2
	local rshift = bit.rshift
	local band = bit.band
	local ldexp = math.ldexp

	local function f16_to_f32(buf)
		return ldexp(buf.mantissa + 1024, buf.exponent - 25) * (1 - buf.sign * 2)
	end

	local function f16_to_f32(bits)
		local sign = 1 - band(rshift(bits, 15), 0x1) * 2
		local exponent = band(rshift(bits, 10), 0x1F)
		local mantissa = band(bits, 0x3FF)
		return sign * ldexp(mantissa + 1024, exponent - 25)
	end

	function Blob:Q4_0(size, blob)
		local byte_size = size * type_size
		blob = ffi.cast("uint8_t*", blob or ffi.cast("uint8_t*", ffi.C.malloc(byte_size)))
		local blob_f16 = ffi.cast(
			[[
			struct {	
				uint16_t mantissa : 10;
				uint16_t exponent : 5;
				uint16_t sign : 1;
			}*
		]],
			blob
		)
		blob_f16 = ffi.cast("uint16_t*", blob)
		assert(byte_size % block_size == 0, "Total size must be a multiple of the block size")
		byte_size = byte_size / block_size
		local floats = ffi.typeof("float[32]")
		local f = floats()
		return setmetatable(
			{
				type = "Q4_0",
				blob = blob,
				size = tonumber(size),
				byte_size = tonumber(byte_size),
				byte_stride = 1,
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
				Get32FloatsFromBlockIndex = function(s, block_index)
					local scale = f16_to_f32(blob_f16[block_index * half_type_size])
					local block_offset = block_index * type_size

					for modIndex = 0, 16 - 1 do
						local byte = blob[block_offset + band(modIndex, half_block_size - 1) + 2]
						f[modIndex] = (band(byte, 0x0F) - 8) * scale
						f[modIndex + 16] = (band(rshift(byte, 4), 0x0F) - 8) * scale
					end

					return f
				end,
			},
			Blob
		)-- :Fill(0, size, 0)
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