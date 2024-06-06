local ffi = require("ffi")
local Blob = require("blob")
local math_exp = math.exp

local Tensor = {}
Tensor.__index = Tensor
ffi.cdef[[
	void *malloc( size_t size );
]]

function Tensor:F32(size, blob)
	return self:new(Blob:F32(size, blob))
end

function Tensor:F64(size, blob)
	return self:new(Blob:F64(size, blob))
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

	function Tensor:Dot2(thisOffset, that, thatOffset, size)
		local result = 0
		thisOffset = thisOffset / 32

		for j = 0, (size / 32) - 1 do
			local floats = self.blob:Get32FloatsFromBlockIndex(thisOffset + j)

			for k = 0, 31 do
				result = result + floats[k] * that:GetFloat(k + thatOffset + j)
			end
		end

		return result
	end

	function Tensor:MatrixVectorMultiply(that, out, dim0, dim1)
		for i = 0, dim0 - 1 do
			out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
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
    for i = thatOffset, thatOffset + size - 1 do
        that:SetFloat(i, self:GetFloat(i - thatOffset + thisOffset))
    end
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

	function Tensor:FillInPlace(thisOffset, size, identity)
		for i = thisOffset, thisOffset + size - 1 do
			self:SetFloat(i, identity)
		end
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
		ss = ss + f*f
	end

	ss = ss / size
	ss = ss + rmsNormEps
	ss = 1.0 / math.sqrt(ss)

	for i = 0, size - 1 do
		self:SetFloat(i, weight:GetFloat(i) * (ss * x:GetFloat(i)))
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