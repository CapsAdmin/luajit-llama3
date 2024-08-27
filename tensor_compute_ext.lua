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
    local ffi = require("ffi")
    local gpu = require("compute.gpu_cuda")
    gpu.init_with_device(0)

    local cache
    do
        cache = ffi.new("float[65536]")
        local function host_f16_to_f32(bits)
            local sign = 1 - bit.band(bit.rshift(bits, 15), 0x1) * 2
            local exponent = bit.band(bit.rshift(bits, 10), 0x1F)
            local mantissa = bit.band(bits, 0x3FF)
            local base = mantissa + 1024
            return sign * math.ldexp(base, exponent - 25)
        end
        for i = 0, 65536 do
            cache[i] = host_f16_to_f32(i)
        end
    end

    local kernel_q40_f32_f32 = gpu.compile_kernel([=[
        #define BLOCK_SIZE 32
        #define HALF_BLOCK_SIZE 16
        #define TYPE_SIZE 18
        #define HALF_TYPE_SIZE 9
        __device__ float f16_to_f32_cache[65536];

        __device__ void decode_float_block(const unsigned char *blob, int block_index, float *f) {
            const unsigned short* blob_f16 = (const unsigned short*)blob;
            
            float scale = f16_to_f32_cache[blob_f16[block_index * HALF_TYPE_SIZE]];
            
            int block_offset = block_index * TYPE_SIZE;
            for (int i = 0; i < 16; i++) {
                unsigned char b = blob[block_offset + (i & (HALF_BLOCK_SIZE - 1)) + 2];

                f[i] = ((b & 0x0F) - 8) * scale;
                f[i+16] = (((b >> 4) & 0x0F) - 8) * scale;
            }
        }

        extern "C" __global__ void kernel_q40_f32_f32(const unsigned char *a, float* b, float* out, int dim0, int dim1) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= dim0)
                return;
                
            __shared__ float float_block[32];
            float result = 0.0f;
            int block_index = (row * dim1) / 32;

            for (int j = 0; j < dim1 / 32; j++) {
                decode_float_block(a, block_index + j, float_block);

                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    result += float_block[k] * b[j*32+k];
                }
            }
            out[row] = result;
        }
    ]=], "kernel_q40_f32_f32", {
        f16_to_f32_cache = {data = cache, size = ffi.sizeof(cache)}
    })


    local kernel_f32_f32_f32 = gpu.compile_kernel([[
        extern "C" __global__ void kernel_f32_f32_f32(float *a, float* b, float* out, int dim0, int dim1) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= dim0)
                return;

            float result = 0.0f;
            int offset = row * dim1;
            for (int j = 0; j < dim1; j++) {
                result += a[offset + j] * b[j];
            }
            out[row] = result;
        }
    ]], "kernel_f32_f32_f32")


    do
        local F32_SIZE = 4
        local SHORT_SIZE = 2
        local ffi = require("ffi")

        local function run_kernel(kernel, a, b, out, dim0, dim1)
            -- this assumes a, b and out have been uploaded and allocated on the gpu
            -- it also assumes a never changes, which in the context of this proejct are the
            -- weights
            gpu.copy_to_device(b.gpu_ptr, b.blob, b.byte_size)

            local thread_count = 1024
            local block_count = math.ceil((dim0 + thread_count - 1) / thread_count)

            local box_dim0 = ffi.new("int[1]", dim0)
            local box_dim1 = ffi.new("int[1]", dim1)
            local args = ffi.new("void*[5]", a.gpu_ptr, b.gpu_ptr, out.gpu_ptr, box_dim0, box_dim1)

            gpu.run_kernel(
                kernel, 
                thread_count, 1, 1, 
                block_count, 1, 1,
                args
            )

            gpu.copy_from_device(out.gpu_ptr, out.blob, dim0 * out.byte_stride)
        end

        Tensor.MatrixVectorMultiplyCPU = Tensor.MatrixVectorMultiply

        function Tensor.MatrixVectorMultiply(a, b, out, dim0, dim1)
            if a.blob.type == "Q4_0" and b.blob.type == "F32" and out.blob.type == "F32" then
                run_kernel(kernel_q40_f32_f32, a.blob, b.blob, out.blob, dim0, dim1)
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