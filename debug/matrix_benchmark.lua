local Tensor = require("tensor")
Tensor:EnableThreadedMatrixVectorMultiply()
local measure = require("debug.measure")

function Tensor:MatrixVectorMultiply(that, out, dim0, dim1)
	for i = 0, dim0 - 1 do
		local result = 0

		for j = 0, dim1 - 1 do
			result = result + self:GetFloat(i * dim1 + j) * that:GetFloat(j)
		end

		out:SetFloat(i, result)
	end
end

local variants = {
	{
		out = Tensor:F32(4096),
		a = Tensor:F32(16777216),
		b = Tensor:F32(4096),
		dim0 = 4096,
		dim1 = 4096,
	},
	{
		out = Tensor:F32(4096),
		a = Tensor:F32(4194304),
		b = Tensor:F32(4096),
		dim0 = 1024,
		dim1 = 4096,
	},
	{
		out = Tensor:F32(14336),
		a = Tensor:F32(58720256),
		b = Tensor:F32(4096),
		dim0 = 14336,
		dim1 = 4096,
	},
	{
		out = Tensor:F32(4096),
		a = Tensor:F32(58720256),
		b = Tensor:F32(14336),
		dim0 = 4096,
		dim1 = 14336,
	},
	{
		out = Tensor:F32(128256),
		a = Tensor:F32(525336576),
		b = Tensor:F32(4096),
		dim0 = 128256,
		dim1 = 4096,
	},
}

local total = 0

for i = 1, 20 do
	if i > 2 then -- jit warmup
	measure("MatrixVectorMultiply") end

	for k, v in ipairs(variants) do
		v.a:MatrixVectorMultiply(v.b, v.out, v.dim0, v.dim1)
	end

	if i > 2 then total = total + measure() end
end

print("avg time: ", total / 20)