local ffi = require("ffi")
ffi.cdef[[
	void *malloc( size_t size );
	void *memcpy(void *dest, const void *src, size_t n);
]]
local math_exp = math.exp
local Tensor = {}
Tensor.__index = Tensor
Tensor.tensors_created = {}

function Tensor.GetAll()
	return Tensor.tensors_created
end

function Tensor:SetName(n)
	self.name = n
	return self
end

function Tensor:__tostring()
	if self.name then return self.name .. "[" .. tostring(self.size) .. "]" end

	return "Tensor[" .. self.size .. "]"
end

do
	function Tensor:Dot(thisOffset, that, thatOffset, size)
		local result = 0

		for j = 0, size - 1 do
			result = result + self:GetFloat(thisOffset + j) * that:GetFloat(thatOffset + j)
		end

		return result
	end

	function Tensor:MatrixVectorMultiply(that, out, dim0, dim1, offset)
		for i = offset or 0, dim0 - 1 do
			local result = 0

			for j = 0, dim1 - 1 do
				result = result + self:GetFloat(i * dim1 + j) * that:GetFloat(j)
			end

			out:SetFloat(i, result)
		end
	end
end

do -- avoid using these except for when debugging
	function Tensor:Reduce(thisOffset, size, seed, reduce_callback)
		local result = seed

		for i = 0, size - 1 do
			result = reduce_callback(result, self:GetFloat(thisOffset + i))
		end

		return result
	end

	function Tensor:MapInPlace(thisOffset, size, F, a, b, c, d)
		local endOffset = thisOffset + size

		for i = thisOffset, endOffset - 1 do
			self:SetFloat(i, F(self:GetFloat(i), i, a, b, c, d))
		end

		return self
	end
end

do
	function Tensor:Sum(thisOffset, size)
		local res = 0

		for i = 0, size - 1 do
			res = res + self:GetFloat(thisOffset + i)
		end

		return res
	end

	local max = math.max

	function Tensor:Max(thisOffset, size)
		local res = 0

		for i = 0, size - 1 do
			res = max(res, self:GetFloat(thisOffset + i))
		end

		return res
	end
end

function Tensor:CopyTo(thisOffset, that, thatOffset, size)
	if self.type == "F32" and that.type == "F32" then
		ffi.C.memcpy(that.blob + thatOffset, self.blob + thisOffset, size * self.byte_stride)
	else
		for i = thatOffset, thatOffset + size - 1 do
			that:SetFloat(i, self:GetFloat(i - thatOffset + thisOffset))
		end
	end
end

function Tensor:FillInPlace(thisOffset, size, identity)
	error("NYI", 2)
end

do
	function Tensor:DivideInPlace(thisOffset, size, value)
		for i = thisOffset, thisOffset + size - 1 do
			self:SetFloat(i, self:GetFloat(i) / value)
		end
	end

	function Tensor:AddTensorInPlaceOffset(thisOffset, that, thatOffset, size)
		for i = thisOffset, thisOffset + size - 1 do
			self:SetFloat(i, self:GetFloat(i) + that:GetFloat(i - thisOffset + thatOffset))
		end
	end

	function Tensor:AddTensorInPlace(that)
		for i = 0, self.size - 1 do
			self:SetFloat(i, self:GetFloat(i) + that:GetFloat(i))
		end
	end

	function Tensor:MultiplyTensorInPlaceOffset(thisOffset, that, thatOffset, size)
		for i = thisOffset, thisOffset + size - 1 do
			self:SetFloat(i, self:GetFloat(i) * that:GetFloat(i - thisOffset + thatOffset))
		end
	end

	function Tensor:MultiplyTensorInPlace(that)
		self:MultiplyTensorInPlaceOffset(0, that, 0, self.size)
	end

	function Tensor:SoftMaxInPlace(thisOffset, size)
		local max_value = self:Max(thisOffset, size)

		for i = thisOffset, thisOffset + size - 1 do
			self:SetFloat(i, math_exp(self:GetFloat(i) - max_value))
		end

		self:DivideInPlace(thisOffset, size, self:Sum(thisOffset, size))
	end
end

function Tensor:SaxpyInPlace(thisOffset, that, thatOffset, size, a)
	for i = 0, size - 1 do
		self:SetFloat(thisOffset + i, a * that:GetFloat(thatOffset + i) + self:GetFloat(thisOffset + i))
	end
end

function Tensor:SigmoidInPlace()
	for i = 0, self.size - 1 do
		local value = self:GetFloat(i)
		self:SetFloat(i, value / (1.0 + math_exp(-value)))
	end
end

function Tensor:RmsNormInPlace(x, weight, size, rmsNormEps)
	local ss = 0

	for i = 0, size - 1 do
		local f = x:GetFloat(i)
		ss = ss + f * f
	end

	ss = ss / size
	ss = ss + rmsNormEps
	ss = 1.0 / math.sqrt(ss)

	for i = 0, size - 1 do
		self:SetFloat(i, weight:GetFloat(i) * (ss * x:GetFloat(i)))
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
		Q8_0 = 2,
		Q6_K = 3,
	}

	-- double lookup
	for i, str in pairs(type_map) do
		type_map[str] = i
	end

	function Tensor:ThreadSerialize()
		return ctype(self.size, type_map[self.type], ffi.cast("void *", self.blob))
	end

	function Tensor:ThreadDeserialize(ptr)
		local data = ffi.cast(ctype_ptr, ptr)

		if not type_map[data.type] then error("unknown type " .. data.type) end

		return Tensor.New(type_map[data.type], data.size, data.blob)
	end
end

function Tensor:UseComputeKernel(backend)
	local build_kernels = require("tensor_kernels")

	for k, v in pairs(build_kernels(backend)) do
		assert(Tensor[k], k .. " is not a function")
		Tensor[k .. "Old"] = Tensor[k]
		Tensor[k] = v
	end
end

local tensor_types = {}

do
	tensor_types.F64 = function(size, blob)
		local stride = ffi.sizeof("double")
		blob = ffi.cast("double*", blob or ffi.cast("double*", ffi.C.malloc(size * stride)))
		return {
			blob = blob,
			size = tonumber(size),
			byte_size = tonumber(size * stride),
			byte_stride = stride,
			SetFloat = function(_, index, val)
				blob[index] = val
			end,
			GetFloat = function(_, index)
				return blob[index]
			end,
		}
	end
	tensor_types.F32 = function(size, blob)
		local stride = 4
		blob = ffi.cast("float*", blob or ffi.cast("float*", ffi.C.malloc(size * stride)))
		return {
			blob = blob,
			size = tonumber(size),
			byte_size = tonumber(size * stride),
			byte_stride = stride,
			SetFloat = function(_, index, val)
				blob[index] = val
			end,
			GetFloat = function(_, index)
				return blob[index]
			end,
			FillInPlace = function(self, thisOffset, size, identity)
				if identity == 0 then
					ffi.fill(self.blob + thisOffset, size * self.byte_stride, 0)
				else
					for i = thisOffset, thisOffset + size - 1 do
						self:SetFloat(i, identity)
					end
				end
			end,
		}
	end

	do
		local ggf = require("gguf")
		local f16_to_f32 = ggf.f16_to_f32
		local block_size = ggf.GGMLTypeMap.Q4_0.block_size
		local half_block_size = block_size / 2
		local type_size = ggf.GGMLTypeMap.Q4_0.type_size
		local half_type_size = type_size / 2
		local rshift = bit.rshift
		local band = bit.band
		tensor_types.Q4_0 = function(size, blob)
			local byte_size = size * type_size
			blob = ffi.cast("uint8_t*", blob or ffi.cast("uint8_t*", ffi.C.malloc(byte_size)))
			local blob_f16 = ffi.cast("uint16_t*", blob)
			assert(byte_size % block_size == 0, "Total size must be a multiple of the block size")
			byte_size = byte_size / block_size
			local floats = ffi.typeof("float[32]")
			local f = floats()
			return {
				blob = blob,
				size = tonumber(size),
				byte_size = tonumber(byte_size),
				byte_stride = 1,
				blob_f16 = blob_f16,
				half_type_size = half_type_size,
				type_size = type_size,
				half_block_size = half_block_size,
				block_size = block_size,
				GetFloat = function(_, index)
					local block_index = rshift(index, 5)
					local block_offset = block_index * type_size
					local scale = f16_to_f32(blob_f16[block_index * half_type_size])
					local modIndex = band(index, block_size - 1)
					local base_offset = block_offset + band(modIndex, half_block_size - 1)
					local shift_amount = rshift(modIndex, 4) * 4
					local quant = band(rshift(blob[2 + base_offset], shift_amount), 0x0F)
					return (quant - 8) * scale
				end,
			}
		end
	end

	do
		local ggf = require("gguf")
		local f16_to_f32 = ggf.f16_to_f32
		local block_size = ggf.GGMLTypeMap.Q6_K.block_size
		local half_block_size = block_size / 2
		local type_size = ggf.GGMLTypeMap.Q6_K.type_size
		local half_type_size = type_size / 2
		local rshift = bit.rshift
		local band = bit.band
		tensor_types.Q6_K = function(size, blob)
			local byte_size = size * type_size
			blob = ffi.cast("uint8_t*", blob or ffi.C.malloc(byte_size))
			local blob_f16 = ffi.cast("uint16_t*", blob)
			assert(byte_size % type_size == 0, "Total size must be a multiple of the block size")
			byte_size = byte_size / type_size
			local floats = ffi.typeof("float[256]") -- Increased to 256 for Q6_K block size
			local f = floats()
			return {
				blob = blob,
				size = tonumber(size),
				byte_size = tonumber(byte_size),
				byte_stride = 1,
				blob_f16 = blob_f16,
				half_type_size = half_type_size,
				block_size = block_size,
				half_block_size = half_block_size,
				type_size = type_size,
				GetFloat = function(_, index)
					local block_index = rshift(index, 8) -- Divide by 256
					local block_offset = block_index * type_size
					local scale = f16_to_f32(blob_f16[block_index * 2])
					local min = f16_to_f32(blob_f16[block_index * 2 + 1])
					-- Calculate position within block
					local modIndex = band(index, 255) -- index % 256
					-- Q6_K uses 6 bits per value
					-- Each byte contains 1.333 values (8/6 bits)
					-- Need to handle bit extraction carefully
					local byte_idx = rshift(modIndex * 6, 3) -- (modIndex * 6) / 8
					local bit_shift = band(modIndex * 6, 7) -- (modIndex * 6) % 8
					-- Extract 6 bits across byte boundary if needed
					local val = band(rshift(blob[block_offset + 4 + byte_idx], bit_shift), 0x3F)

					if bit_shift > 2 then
						-- Need some bits from next byte
						val = band(val + lshift(blob[block_offset + 5 + byte_idx], 8 - bit_shift), 0x3F)
					end

					-- Dequantize: val * scale + min
					return val * scale + min
				end,
			}
		end
	end

	do
		local ggf = require("gguf")
		local block_size = 32 -- Standard block size
		local type_size = block_size -- For Q8_0, type_size equals block_size since each value is 1 byte
		local rshift = bit.rshift
		local band = bit.band
		tensor_types.Q8_0 = function(size, blob)
			local byte_size = size
			blob = ffi.cast("int8_t*", blob or ffi.C.malloc(byte_size))
			assert(byte_size % block_size == 0, "Total size must be a multiple of the block size")
			byte_size = byte_size / block_size
			local floats = ffi.typeof("float[32]")
			local f = floats()
			return {
				blob = blob,
				size = tonumber(size),
				byte_size = tonumber(byte_size),
				byte_stride = 1,
				-- Get a single float value at the given index
				GetFloat = function(_, index)
					-- Q8_0 values are direct 8-bit integers, no scaling needed
					return tonumber(blob[index])
				end,
			}
		end
	end
end

function Tensor.New(typ, size, blob)
	if not tensor_types[typ] then error("NYI tensor type: " .. tostring(typ), 2) end

	local t = setmetatable(tensor_types[typ](size, blob), Tensor)
	t.type = typ
	table.insert(Tensor.tensors_created, t)
	return t
end

return Tensor
