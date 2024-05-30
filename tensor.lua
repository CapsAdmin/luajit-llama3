local ffi = require("ffi")
local Tensor = {}
Tensor.__index = Tensor
ffi.cdef[[
	void *malloc( size_t size );
]]

local function valid_number(num, self, i)
	if num and num ~= inf and num ~= ninf and (num >= 0 or num <= 0) then
		return num
	end

	error(
		tostring(self) .. ".blob contains an invalid number at index " .. i .. ": " .. tostring(num)
	)
	return num
end

function Tensor:new(size, blob, blob_ref, get_float, set_float)
	local t = setmetatable({}, Tensor)
	assert(type(size) == "number" or type(size) == "cdata") -- ULL
	assert(type(blob) == "cdata")
	assert(blob_ref)
	assert(type(set_float) == "function")
	assert(type(get_float) == "function")
	t.blob = blob
	t.size = tonumber(size)
	t.SetFloat = set_float
	t.GetFloat = get_float
	t.blob_ref = blob_ref -- need to keep this around for gc
	return t
end

function Tensor:SetName(n)
	self.name = n
	return self
end

function Tensor:__tostring()
	if self.name then return self.name .. "[" .. tostring(self.size) .. "]" end

	return "Tensor[" .. self.size .. "]"
end

local ggf = require("gguf")

do
	local function get_float(self, index)
		return self.blob[index]
	end

	local function set_float(self, index, val)
		self.blob[index] = val
	end

	Tensor.GetF32 = get_float
	Tensor.SetF32 = set_float

	function Tensor:F32(size, blob)
		local blob_ref = blob

		if not blob then
			blob = ffi.cast("float*", ffi.C.malloc(size * 4))
			--ffi.fill(blob, size * 4, 0)
			blob_ref = blob
		end

		return Tensor:new(size, ffi.cast("float *", blob), blob_ref, get_float, set_float)
	end
end

do
	local block_size = ggf.GGMLTypeMap.Q4_0.blockSize
	local half_block_size = block_size / 2
	local type_size = ggf.GGMLTypeMap.Q4_0.typeSize
	local rshift = bit.rshift
	local band = bit.band
	local ldexp = math.ldexp
	local floor = math.floor

	-- f16_to_f32 is not accurate because we don't need to handle nan, inf, etc
	local function f16_to_f32(bits)
		local sign = 1 - band(rshift(bits, 15), 0x1) * 2
		local exponent = band(rshift(bits, 10), 0x1F)
		local mantissa = band(bits, 0x3FF)
		return sign * ldexp(mantissa + 1024, exponent - 25)
	end

	local function get_float(self, index)
		local block_index = floor(index / block_size)
		local block_offset = floor(block_index * type_size)
		local scale = f16_to_f32(self.blob_f16[block_offset / 2])
		-- Calculate the shift amount using bitwise operations to avoid branches
		local modIndex = band(index, block_size - 1)
		local base_offset = block_offset + 2 + band(modIndex, half_block_size - 1)
		local shift_amount = rshift(modIndex, 4) * 4
		local quant = band(rshift(self.blob[base_offset], shift_amount), 0x0F)
		return (quant - 8) * scale
	end

	local function set_float(self, index, value)
		assert(index >= 0)
		assert(index < self.size)
		error("NYI", 2)
	end

	Tensor.GetQ4_0 = get_float
	Tensor.SetQ4_0 = set_float

	function Tensor:Q4_0(size, blob)
		local blob_ref = blob

		if not blob then
			blob = ffi.cast("uint8_t*", ffi.C.malloc(size))
			ffi.fill(blob, size, 0)
			blob_ref = blob
		end

		local t = Tensor:new(size, ffi.cast("uint8_t *", blob), blob_ref, get_float, set_float)
		t.blob_f16 = ffi.cast("uint16_t*", t.blob)
		return t
	end
end

function Tensor:GetFloatVector(index, value)
	error("NYI")
end

function Tensor:Dot(thisOffset, that, thatOffset, size)
	local result = 0

	for j = 0, size - 1 do
		result = result + self:GetFloat(thisOffset + j) * that:GetFloat(thatOffset + j)
	end

	return result
end

do
	function Tensor:MatrixDotProduct(that, out, dim0, dim1)
		for i = 0, dim0 - 1 do
			out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
		end
	end

	function Tensor.EnableThreadedMatrixDotProduct() 
		local ok, err = pcall(function()

			local build_parallel_for = require("threads")
			local parallel_for = build_parallel_for(function(dim1, out, self, that, thread_data)
				local i = thread_data
				out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
			end, {"double", "@tensor", "@tensor", "@tensor"}, 64)
			
			local done = {}
			function Tensor:MatrixDotProduct(that, out, dim0, dim1)
				parallel_for(dim0, dim1, out, self, that)
			end
		end)

		if not ok then print("threading can't be enabled: " .. err) end
	end
end

do
	function Tensor:Reduce(thisOffset, size, seed, reduce_callback)
		local result = seed

		for i = 0, size - 1 do
			result = reduce_callback(result, self:GetFloat(thisOffset + i))
		end

		return result
	end

	do
		local function F(r, f)
			return r + f
		end

		function Tensor:Sum(thisOffset, size)
			return self:Reduce(thisOffset, size, 0, F)
		end
	end

	do
		local max = math.max

		function Tensor:Max(thisOffset, size)
			return self:Reduce(thisOffset, size, 0, max)
		end
	end
end

do
	local function F(value, index, self, thatOffset, thisOffset)
		return self:GetFloat(index - thatOffset + thisOffset)
	end

	function Tensor:CopyTo(thisOffset, that, thatOffset, size)
		return that:MapInPlace(thatOffset, size, F, self, thatOffset, thisOffset)
	end
end

do
	function Tensor:MapInPlace(thisOffset, size, F, a, b, c, d)
		local endOffset = thisOffset + size

		for i = thisOffset, endOffset - 1 do
			self:SetFloat(i, F(self:GetFloat(i), i, a, b, c, d))
		end

		return self
	end

	do
		local function F(value, index, div)
			return value / div
		end

		function Tensor:DivideInPlace(thisOffset, size, value)
			return self:MapInPlace(thisOffset, size, F, value)
		end
	end

	do
		local function F(value, index, that, thisOffset, thatOffset)
			return value + that:GetFloat(index - thisOffset + thatOffset)
		end

		function Tensor:AddTensorInPlaceOffset(thisOffset, that, thatOffset, size)
			return self:MapInPlace(thisOffset, size, F, that, thisOffset, thatOffset)
		end
	end

	function Tensor:AddTensorInPlace(that)
		return self:AddTensorInPlaceOffset(0, that, 0, self.size)
	end

	do
		local function F(value, index, that, thisOffset, thatOffset)
			return value * that:GetFloat(index - thisOffset + thatOffset)
		end

		function Tensor:MultiplyTensorInPlaceOffset(thisOffset, that, thatOffset, size)
			return self:MapInPlace(thisOffset, size, F, that, thisOffset, thatOffset)
		end
	end

	function Tensor:MultiplyTensorInPlace(that)
		return self:MultiplyTensorInPlaceOffset(0, that, 0, self.size)
	end

	do
		local function F(value, index, identity)
			return identity
		end

		function Tensor:FillInPlace(thisOffset, size, identity)
			return self:MapInPlace(thisOffset, size, F, identity)
		end
	end

	do
		local exp = math.exp

		local function F(num, index, max_value)
			return exp(num - max_value)
		end

		function Tensor:SoftMaxInPlace(thisOffset, size)
			self:MapInPlace(thisOffset, size, F, self:Max(thisOffset, size))
			return self:DivideInPlace(thisOffset, size, self:Sum(thisOffset, size))
		end
	end
end

function Tensor:SaxyInPlace(thisOffset, that, thatOffset, size, a)
	for i = 0, size - 1 do
		self:SetFloat(thisOffset + i, a * that:GetFloat(thatOffset + i) + self:GetFloat(thisOffset + i))
	end

	return self
end

do
	local function F1(acc, xi)
		return acc + xi * xi
	end

	local function F2(value, index, weight, ss, x)
		return weight:GetFloat(index) * (ss * x:GetFloat(index))
	end

	function Tensor:RmsNormInPlace(x, weight, size, rmsNormEps)
		local ss = x:Reduce(0, size, 0, F1)
		ss = ss / size
		ss = ss + rmsNormEps
		ss = 1.0 / math.sqrt(ss)
		self:MapInPlace(0, size, F2, weight, ss, x)
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

	function Tensor:ThreadSerialize()
		local ct = ctype(
			self.size,
			self.GetFloat == Tensor.GetF32 and 0 or 1,
			ffi.cast("void *", self.blob)
		)
		return ct
	end

	function Tensor:ThreadDeserialize(ptr)
		local data = ffi.cast(ctype_ptr, ptr)

		if data.type == 0 then
			return setmetatable(
				{
					size = data.size,
					blob = ffi.cast("float*", data.blob),
					GetFloat = Tensor.GetF32,
					SetFloat = Tensor.SetF32,
				},
				Tensor
			)
		else
			return setmetatable(
				{
					size = data.size,
					blob = ffi.cast("uint8_t*", data.blob),
					blob_f16 = ffi.cast("uint16_t*", data.blob),
					GetFloat = Tensor.GetQ4_0,
					SetFloat = Tensor.SetQ4_0,
				},
				Tensor
			)
		end
	end
end

return Tensor