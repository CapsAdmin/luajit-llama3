local Tensor = require("tensor")

local function use_pthreads()
    local build_parallel_for = require("compute.cpu_pthreads")
    local parallel_for = build_parallel_for(function(dim1, out, self, that, thread_data)	
        local i = thread_data
        out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
    end, {"double", "@tensor", "@tensor", "@tensor"}, 64)
    
    local done = {}
    function Tensor:MatrixVectorMultiply(that, out, dim0, dim1)
        parallel_for(dim0, dim1, out, self, that)
    end
end

local function use_cuda()
    local gpu = require("compute.gpu_cuda")

    gpu.cuda.cuInit(0)
    local context = gpu.create_context()
    gpu.cuda.cuCtxSetCurrent(context)
    
    local kernel_f32_q4_0_f32 = gpu.compile_ptx(gpu.source_to_ptx([[
        #define BLOCK_SIZE 32
        #define HALF_BLOCK_SIZE 16
        #define TYPE_SIZE 18
        #define HALF_TYPE_SIZE 9

        __device__ float f16_to_f32(unsigned short bits) {
            int sign = 1 - ((bits >> 15) & 0x1) * 2;
            int exponent = (bits >> 10) & 0x1F;
            int mantissa = bits & 0x3FF;
            int base = (float)(mantissa + 1024);
            return (float)sign * (float)ldexpf(base, exponent - 25);
        }

        __device__ float get_float(const unsigned char *blob, int index) {
            const unsigned short* blob_f16 = (const unsigned short*)blob;
            int block_index = index >> 5;					
            int block_offset = block_index * TYPE_SIZE;
            float scale = f16_to_f32(blob_f16[block_index * HALF_TYPE_SIZE]);
            int modIndex = index & (BLOCK_SIZE - 1);
            int base_offset = block_offset + (modIndex & (HALF_BLOCK_SIZE - 1));
            int shift_amount = (modIndex >> 4) * 4; 
            int quant = (blob[2 + base_offset] >> shift_amount) & 0x0F;

            return (quant - 8) * scale;
        }

        extern "C" __global__ void kernel_f32_q4_0_f32(const unsigned char *self, float* that, float* out, int dim0, int dim1) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < dim0) { 
                float result = 0.0f;
                for (int j = 0; j < dim1; j++) {
                    result += get_float(self, row * dim1 + j) * that[j];
                }
                out[row] = result;
            }
        }
    ]]), "kernel_f32_q4_0_f32")


    local kernel_f32_f32_f32 = gpu.compile_ptx(gpu.source_to_ptx([[
        extern "C" __global__ void kernel_f32_f32_f32(float *self, float* that, float* out, int dim0, int dim1) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < dim0) { 
                float result = 0.0f;
                for (int j = 0; j < dim1; j++) {
                    result += self[row * dim1 + j] * that[j];
                }
                out[row] = result;
            }
        }
    ]]), "kernel_f32_f32_f32")


    do
        local F32_SIZE = 4
        local SHORT_SIZE = 2
        local ffi = require("ffi")

        local cache = {}
        local function cached_gpu_allocate(key, size)
            cache[key] = cache[key] or {}

            if not cache[key][size] then
                cache[key][size] = gpu.allocate(size)
            end

            return cache[key][size]
        end

        local function optimalThreadsPerBlock(dim0)
            local baseBlockSize = 256  -- Starting point for block size
            if dim0 < 512 then
                return 128  -- Use smaller blocks for small dimensions
            elseif dim0 > 10000 then
                return 512  -- Increase block size for very large dimensions
            else
                return baseBlockSize
            end
        end

        local measure = require("debug.measure")

        local function run_kernel(kernel, self, that, out, dim0, dim1)

            local self_gpu = cached_gpu_allocate("self", self.byte_size)
            local that_gpu = cached_gpu_allocate("that", that.byte_size)
            local out_gpu = cached_gpu_allocate("out", out.byte_size)

            gpu.cuda.cuMemcpyHtoD(self_gpu[0], self.blob, self.byte_size)
            gpu.cuda.cuMemcpyHtoD(that_gpu[0], that.blob, that.byte_size)
                                
            local threadsPerBlock = optimalThreadsPerBlock(dim0)
            local numBlocks = math.ceil(dim0 / threadsPerBlock)

            local box_sx = ffi.new("int[1]", dim0)
            local box_sy = ffi.new("int[1]", dim1)
            local args = ffi.new("void*[5]", self_gpu, that_gpu, out_gpu, box_sx, box_sy)
            
            gpu.cuda.cuLaunchKernel(
                kernel, 
                threadsPerBlock, 1, 1, 
                numBlocks, 1, 1, 
                0, nil, args, nil
            )

            gpu.cuda.cuMemcpyDtoH(out.blob, out_gpu[0], out.byte_size)
        end

        Tensor.MatrixVectorMultiplyCPU = Tensor.MatrixVectorMultiply

        function Tensor:MatrixVectorMultiply(that, out, dim0, dim1)
            self:MatrixVectorMultiplyCPU(that, out, dim0, dim1)
            local expected = out
            local out = Tensor:F32(out.size)
            
            if self.blob.type == "Q4_0" and that.blob.type == "F32" and out.blob.type == "F32" then
                run_kernel(kernel_f32_q4_0_f32, self.blob, that.blob, out.blob, dim0, dim1)
            elseif self.blob.type == "F32" and that.blob.type == "F32" and out.blob.type == "F32" then
                run_kernel(kernel_f32_f32_f32, self.blob, that.blob, out.blob, dim0, dim1)
            else
                error("NYI")
            end

            for i = 0, out.size - 1 do
                local a, b = out:GetFloat(i), expected:GetFloat(i)
        
                if (a - b) > 0.01 then
                    print(self, "failed at index", i)
                    print(a, " ~= ", b, (a - b))
                    error("")
                    break
                end
            end
        end
    end
end

return {
    use_cuda = use_cuda,
    use_pthreads = use_pthreads,
}