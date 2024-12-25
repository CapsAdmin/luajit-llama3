local Tensor = require("tensor")
Tensor:UseComputeKernel("pthreads")
local measure = require("debug.measure")

local variants = {
	{
		out = Tensor.New("F32", 4096),
		a = Tensor.New("F32", 16777216),
		b = Tensor.New("F32", 4096),
		dim0 = 4096,
		dim1 = 4096,
	},
	{
		out = Tensor.New("F32", 4096),
		a = Tensor.New("F32", 4194304),
		b = Tensor.New("F32", 4096),
		dim0 = 1024,
		dim1 = 4096,
	},
	{
		out = Tensor.New("F32", 14336),
		a = Tensor.New("F32", 58720256),
		b = Tensor.New("F32", 4096),
		dim0 = 14336,
		dim1 = 4096,
	},
	{
		out = Tensor.New("F32", 4096),
		a = Tensor.New("F32", 58720256),
		b = Tensor.New("F32", 14336),
		dim0 = 4096,
		dim1 = 14336,
	},
	{
		out = Tensor.New("F32", 128256),
		a = Tensor.New("F32", 525336576),
		b = Tensor.New("F32", 4096),
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