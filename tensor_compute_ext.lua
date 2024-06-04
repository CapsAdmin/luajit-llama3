local Tensor = require("tensor")

local function use_pthreads()
    local pthreads = require("compute.cpu_pthreads")
    local threaded_for = pthreads.threaded_for(function(dim1, out, self, that, thread_data)	
        local i = thread_data
        out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
    end, {"double", "@tensor", "@tensor", "@tensor"}, pthreads.get_cpu_threads())
    
    local done = {}
    function Tensor:MatrixVectorMultiply(that, out, dim0, dim1)
        threaded_for(dim0, dim1, out, self, that)
    end
end

local function use_cuda()
    local gpu = require("compute.gpu_cuda")
    gpu.init_with_device(0)

    local kernel_f32_q4_0_f32 = gpu.compile_kernel([[
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

        extern "C" __global__ void kernel_f32_q4_0_f32(const unsigned char *a, float* b, float* out, int dim0, int dim1) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= dim0)
                return;
                
            float result = 0.0f;
            for (int j = 0; j < dim1; j++) {
                result += get_float(a, row * dim1 + j) * b[j];
            }
            out[row] = result;
        }
    ]], "kernel_f32_q4_0_f32")


    local kernel_f32_f32_f32 = gpu.compile_kernel([[
        extern "C" __global__ void kernel_f32_f32_f32(float *a, float* b, float* out, int dim0, int dim1) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= dim0)
                return;

            float result = 0.0f;
            for (int j = 0; j < dim1; j++) {
                result += a[row * dim1 + j] * b[j];
            }
            out[row] = result;
        }
    ]], "kernel_f32_f32_f32")


    do
        local F32_SIZE = 4
        local SHORT_SIZE = 2
        local ffi = require("ffi")

        local cache = {}
        local function cached_gpu_allocate(key, size, blob)
            cache[key] = cache[key] or {}

            if blob then
                if not cache[key][blob] then
                    cache[key][blob] = gpu.allocate_on_device(size, blob.blob)
                end
                return cache[key][blob]
            end

            if not cache[key][size] then
                cache[key][size] = gpu.allocate_on_device(size)
            end

            return cache[key][size]
        end

        local function run_kernel(kernel, a, b, out, dim0, dim1)

            local a_gpu = cached_gpu_allocate("a", a.byte_size, a) -- lazily allocate the weights on gpu, assuming a is always the llama weights
            local b_gpu = cached_gpu_allocate("b", b.byte_size)
            local out_gpu = cached_gpu_allocate("out", out.byte_size)

            --gpu.copy_to_device(a_gpu, a.blob, a.byte_size) -- no need, the weights never change
            gpu.copy_to_device(b_gpu, b.blob, b.byte_size)

            local thread_count = 1024
            local block_count = math.ceil((dim0 + thread_count - 1) / thread_count)

            local box_dim0 = ffi.new("int[1]", dim0)
            local box_dim1 = ffi.new("int[1]", dim1)
            local args = ffi.new("void*[5]", a_gpu, b_gpu, out_gpu, box_dim0, box_dim1)

            gpu.run_kernel(
                kernel, 
                thread_count, 1, 1, 
                block_count, 1, 1,
                args
            )

            gpu.copy_from_device(out_gpu, out.blob, dim0 * out.byte_stride)
        end

        Tensor.MatrixVectorMultiplyCPU = Tensor.MatrixVectorMultiply

        function Tensor.MatrixVectorMultiply(a, b, out, dim0, dim1)
            if a.blob.type == "Q4_0" and b.blob.type == "F32" and out.blob.type == "F32" then
                run_kernel(kernel_f32_q4_0_f32, a.blob, b.blob, out.blob, dim0, dim1)
            elseif a.blob.type == "F32" and b.blob.type == "F32" and out.blob.type == "F32" then
                run_kernel(kernel_f32_f32_f32, a.blob, b.blob, out.blob, dim0, dim1)
            else
                error("NYI")
            end
        end
    end
end

return {
    use_cuda = use_cuda,
    use_pthreads = use_pthreads,
}