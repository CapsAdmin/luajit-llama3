local ffi = require("ffi")
local gpu = require("gpu_cuda")
local Tensor = require("tensor")

local function runKernel(ptx, name, a, b, out, sx, sy, numBlocks, threadsPerBlock)
    gpu.cuda.cuInit(0)

    local context = gpu.create_context()
    gpu.cuda.cuCtxSetCurrent(context)

    local kernel = gpu.compile_ptx(ptx, name)

    local F32_SIZE = 4

    local device_a = gpu.allocate(a.size*F32_SIZE, a.blob.blob)
    local device_b = gpu.allocate(b.size*F32_SIZE, b.blob.blob)

    local device_out = gpu.allocate(out.size*F32_SIZE)

    local box_sx = ffi.new("int[1]", sx)
    local box_sy = ffi.new("int[1]", sy)
    local args = ffi.new("void*[5]", device_a, device_b, device_out, box_sx, box_sy)
    gpu.cuda.cuLaunchKernel(
        kernel, 
        threadsPerBlock, 1, 1, 
        numBlocks, 1, 1, 

        0, nil, args, nil
    )

    gpu.cuda.cuMemcpyDtoH(out.blob.blob, device_out[0], out.size*F32_SIZE)

    gpu.cuda.cuMemFree(device_a[0])
    gpu.cuda.cuMemFree(device_b[0])
    gpu.cuda.cuMemFree(device_out[0])

    gpu.cuda.cuCtxDestroy(context)
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

math.randomseed(1337)
for k, v in ipairs(variants) do
    v.a:MapInPlace(0, v.a.size, function(v, i) return math.random()*2-1 end)
    v.b:MapInPlace(0, v.b.size, function(v, i) return math.random()*2-1 end)
end

for k, v in ipairs(variants) do
    v.a:MatrixVectorMultiply(v.b, v.out, v.dim0, v.dim1)
    local out = Tensor:F32(v.out.size)
    
    local threadsPerBlock = 256
    local numBlocks = math.ceil(v.dim0 / threadsPerBlock)
    runKernel(gpu.source_to_ptx([[
        extern "C" __global__ void MatrixVectorMultiply(float* self, float* that, float* out, int dim0, int dim1) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < dim0) {  // Ensure we do not go out of bounds
                float result = 0.0f;
                for (int j = 0; j < dim1; j++) {
                    result += self[row * dim1 + j] * that[j];
                }
                out[row] = result;
            }
        }
    ]]), "MatrixVectorMultiply", v.a, v.b, out, v.dim0, v.dim1, numBlocks, threadsPerBlock)


    for i = 0, out.size - 1 do
        local a, b = out:GetFloat(i), v.out:GetFloat(i)

        if (a - b) > 0.01 then
            print(a, " ~= ", b, (a - b))
            break
        end
    end
end