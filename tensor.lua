local ffi = require("ffi")
local Tensor = {}
Tensor.__index = Tensor

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
		assert(index >= 0)
		assert(index < self.size)
		return self.blob[index]
	end

	local function set_float(self, index, val)
		assert(index >= 0)
		assert(index < self.size)
		self.blob[index] = val
	end

	function Tensor:F32(size, blob)
		local blob_ref = blob

		if not blob then
			blob = ffi.new("float[?]", size)
			blob_ref = blob
		end

		return Tensor:new(size, ffi.cast("float *", blob), blob_ref, get_float, set_float)
	end
end

do
	local block_size = ggf.GGMLTypeMap.Q4_0.blockSize
	local type_size = ggf.GGMLTypeMap.Q4_0.typeSize
	local FLOAT16 = 2
	local rshift = bit.rshift
	local band = bit.band
	local ldexp = math.ldexp
	local INF = math.huge
	local NAN = math.huge / math.huge
	local floor = math.floor

	local function f16_to_f32(bits)
		local sign = bit.band(bit.rshift(bits, 15), 0x1)
		local exponent = bit.band(bit.rshift(bits, 10), 0x1F)
		local mantissa = bit.band(bits, 0x3FF)
		sign = sign == 1 and -1 or 1

		if exponent == 0 then
			if mantissa == 0 then
				return sign * 0.0
			else
				-- Subnormal number
				return sign * ldexp(mantissa, -24)
			end
		elseif exponent == 31 then
			if mantissa == 0 then
				return sign * math.huge
			else
				return sign * (math.huge - math.huge) -- NaN
			end
		else
			return sign * ldexp(mantissa + 1024, exponent - 25)
		end
	end

	local function get_float(self, index)
		assert(index >= 0)
		assert(index < self.size)
		local block_index = floor(index / block_size)
		local block_offset = floor(block_index * type_size)
		local scale = f16_to_f32(self.blob_f16[block_offset / 2])
		local quant
		local modIndex = index % block_size

		if modIndex < block_size / 2 then
			quant = band(self.blob[block_offset + 2 + modIndex], 0x0F)
		else
			quant = band(rshift(self.blob[block_offset + 2 + modIndex - block_size / 2], 4), 0x0F)
		end

		quant = quant - 8
		quant = quant * scale
		return quant
	end

	local function set_float(self, index, value)
		assert(index >= 0)
		assert(index < self.size)
		error("NYI", 2)
	end

	function Tensor:Q4_0(size, blob)
		local blob_ref = blob

		if not blob then
			blob = ffi.new("uint8_t[?]", size)
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

function Tensor:ScalarDot(thisOffset, that, thatOffset, size)
	local result = 0

	for j = 0, size - 1 do
		local a = self:GetFloat(thisOffset + j)
		local b = that:GetFloat(thatOffset + j)
		result = result + a * b
	end

	return result
end

function Tensor:Dot(thisOffset, that, thatOffset, size)
	return self.ScalarDot(self, thisOffset, that, thatOffset, size)
end

function Tensor:MatMul(that, out, dim0, dim1)
	for i = 0, dim0 - 1 do
		out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
	end
end

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

do
	local function F(value, index, self, thatOffset, thisOffset)
		return self:GetFloat(index - thatOffset + thisOffset)
	end

	function Tensor:CopyTo(thisOffset, that, thatOffset, size)
		return that:MapWithIndexInPlace(thatOffset, size, F, self, thatOffset, thisOffset)
	end
end

function Tensor:MapInPlace(thisOffset, size, F, ...)
	local endIndex = thisOffset + size

	for i = thisOffset, endIndex - 1 do
		self:SetFloat(i, F(self:GetFloat(i), ...))
	end

	return self
end

do
	local function F(f, value)
		return f / value
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
		return self:MapWithIndexInPlace(thisOffset, size, F, that, thisOffset, thatOffset)
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
		return self:MapWithIndexInPlace(thisOffset, size, F, that, thisOffset, thatOffset)
	end
end

function Tensor:MultiplyTensorInPlace(that)
	return self:MultiplyTensorInPlaceOffset(0, that, 0, self.size)
end

do
	local function F(f, value)
		return value
	end

	function Tensor:FillInPlace(thisOffset, size, value)
		return self:MapInPlace(thisOffset, size, F, value)
	end
end

function Tensor:SaxyInPlace(thisOffset, that, thatOffset, size, a)
	for i = 0, size - 1 do
		self:SetFloat(thisOffset + i, a * that:GetFloat(thatOffset + i) + this:GetFloat(thisOffset + i))
	end

	return self
end

do
	local exp = math.exp

	local function F(num, maxVal)
		return exp(num - maxVal)
	end

	function Tensor:SoftMaxInPlace(thisOffset, size)
		self:MapInPlace(thisOffset, size, F, self:Max(thisOffset, size))
		return self:DivideInPlace(thisOffset, size, self:Sum(thisOffset, size))
	end
end

function Tensor:MapWithIndexInPlace(thisOffset, size, F, ...)
	local endOffset = thisOffset + size

	for i = thisOffset, endOffset - 1 do
		self:SetFloat(i, F(self:GetFloat(i), i, ...))
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
end

return Tensor