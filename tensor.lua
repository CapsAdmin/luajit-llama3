local ffi = require("ffi")
local Blob = require("blob")
local Tensor = {}
Tensor.__index = Tensor
ffi.cdef[[
	void *malloc( size_t size );
]]

function Tensor:F32(size, blob)
	return self:new(Blob:F32(size, blob))
end

function Tensor:Q4_0(size, blob)
	return self:new(Blob:Q4_0(size, blob))
end

function Tensor:new(blob)
	local t = setmetatable({}, Tensor)
	t.blob = blob
	t.size = blob.size
	return t
end

function Tensor:GetFloat(i)
	return self.blob:GetFloat(i)
end

function Tensor:SetFloat(i, v)
	return self.blob:SetFloat(i, v)
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

	function Tensor:MatrixVectorMultiply(that, out, dim0, dim1)
		for i = 0, dim0 - 1 do
			out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
		end
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
	function Tensor:ThreadSerialize()
		return self.blob:ThreadSerialize()
	end

	function Tensor:ThreadDeserialize(ptr)
		return Tensor:new(Blob:ThreadDeserialize(ptr))
	end
end

return Tensor