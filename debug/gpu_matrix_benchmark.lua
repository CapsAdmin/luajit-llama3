local Tensor = require("tensor"):UseComputeKernel("gpu")
local measure = require("debug.measure")

function Tensor.MatrixVectorMultiplyCPU(a, b, out, dim0, dim1)
	for i = 0, dim0 - 1 do
		local result = 0

		for j = 0, dim1 - 1 do
			result = result + a:GetFloat(i * dim1 + j) * b:GetFloat(j)
		end

		out:SetFloat(i, result)
	end
end

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

do
	-- upload and preallocate tensor memory for better performance and vram usage
	local total_size = 0
	local gpu = require("compute.gpu_cuda")
	measure("uploading tensors to gpu")
	local size_map = {}

	for _, tensors in ipairs(variants) do
		for key, tensor in pairs(tensors) do
			if getmetatable(tensor) == Tensor then
				if tensor.name and tensor.name:find(".weight") then
					-- weight tensors are static
					tensor.gpu_ptr = gpu.allocate_on_device(tensor.byte_size, tensor.blob)
					total_size = total_size + tensor.byte_size
				else
					-- state tensors are dynamic and are uploaded on each Tensor.MatrixVectorMultiply call
					-- so we can allocate and share memory for each byte size
					size_map[tensor.byte_size] = size_map[tensor.byte_size] or {}
					table.insert(size_map[tensor.byte_size], tensor)
				end
			end
		end
	end

	for byte_size, tensors in pairs(size_map) do
		local gpu_ptr = gpu.allocate_on_device(byte_size)

		for _, tensor in ipairs(tensors) do
			tensor.gpu_ptr = gpu_ptr
		end

		total_size = total_size + byte_size
	end

	measure()
	print(string.format("%.2fgb tensors allocated on GPU", total_size / 1024 / 1024 / 1024))
	gpu.dump_gpu_stats()
end

-- zero fill, otherwise GetFloat might return nil or nan
for i,v in ipairs(variants) do
	v.out:FillInPlace(0, v.out.size, 0)
	v.a:FillInPlace(0, v.out.size, 0)
	v.b:FillInPlace(0, v.out.size, 0)
end

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