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

function Tensor:MatMul(that, out, dim0, dim1)
	for i = 0, dim0 - 1 do
		out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
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
		self:SetFloat(thisOffset + i, a * that:GetFloat(thatOffset + i) + this:GetFloat(thisOffset + i))
	end

	return self
end

do -- some tests
	local ggf = require("gguf")
	local t = Tensor:F32(10)

	for i = 0, 10 - 1 do
		t:SetFloat(i, i)
	end

	for i = 0, 10 - 1 do
		assert(t:GetFloat(i) == i)
	end

	assert(t:Sum(0, t.size) == 45)
	assert(t:Max(0, t.size) == 9)
	local t2 = Tensor:F32(10)
	t:CopyTo(0, t2, 0, t.size)

	for i = 0, 10 - 1 do
		assert(t2:GetFloat(i) == i)
	end

	for i = 0, 10 - 1 do
		t2:SetFloat(i, 0)
	end

	t:CopyTo(5, t2, 0, 5)

	for i = 5, 9 do
		assert(t2:GetFloat(i - 5) == i)
	end

	do
		local size = 10
		local t1 = Tensor:F32(size)
		local t2 = Tensor:F32(size)

		for i = 0, size - 1 do
			t1:SetFloat(i, i)
			t2:SetFloat(i, i * 2)
		end

		local dot_product = t1:Dot(0, t2, 0, size)
		local expected_dot_product = 0

		for i = 0, size - 1 do
			expected_dot_product = expected_dot_product + i * (i * 2)
		end

		assert(expected_dot_product == dot_product)
	end

	do -- test for MatMul
		local size = 3
		local t1 = Tensor:F32(size * size)
		local t2 = Tensor:F32(size * size)
		local out = Tensor:F32(size)
		t1:SetFloat(0, 1)
		t1:SetFloat(1, 2)
		t1:SetFloat(2, 3)
		t1:SetFloat(3, 4)
		t1:SetFloat(4, 5)
		t1:SetFloat(5, 6)
		t1:SetFloat(6, 7)
		t1:SetFloat(7, 8)
		t1:SetFloat(8, 9)
		t2:SetFloat(0, 1)
		t2:SetFloat(1, 0)
		t2:SetFloat(2, 0)
		t2:SetFloat(3, 0)
		t2:SetFloat(4, 1)
		t2:SetFloat(5, 0)
		t2:SetFloat(6, 0)
		t2:SetFloat(7, 0)
		t2:SetFloat(8, 1)
		t1:MatMul(t2, out, size, size)
		assert(out:GetFloat(0) == 1)
		assert(out:GetFloat(1) == 4)
		assert(out:GetFloat(2) == 7)
	end
end

return Tensor