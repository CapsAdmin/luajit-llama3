local ffi = require("ffi")

local function MatrixVectorMultiply(a, b, out, dim0, dim1)
	for i = 0, dim0 - 1 do
		local result = 0

		for j = 0, dim1 - 1 do
			result = result + a[i * dim1 + j] * b[j]
		end

		out[i] = result
	end
end

local Tensor = require("tensor")
local out = ffi.new("float[?]", 4096)
local a = ffi.new("float[?]", 58720256)
local b = ffi.new("float[?]", 14336)
local dim0 = 4096
local dim1 = 14336

local total = 0
for i = 1, 20 do
    local time = os.clock()
    MatrixVectorMultiply(a, b, out, dim0, dim1)
    total = total + os.clock() - time
end

print("avg time: ", total / 20)