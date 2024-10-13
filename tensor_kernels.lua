return function(backend)
    assert(backend == "lua" or backend == "pthreads" or backend == "cuda", "backend must be lua, pthreads or cuda")

    if backend == "lua" then

        local rshift = bit.rshift
        local band = bit.band

        local function kernel_q40_f32_f32(a, b, out, dim0, dim1, offset, cached_f16_to_f32, blob_f16, half_type_size, type_size, half_block_size)
            for row = offset or 0, dim0 - 1 do
                local result = 0
                
                local block_index = (row * dim1) / 32
                for j = 0, (dim1 / 32) - 1 do
                    local scale = cached_f16_to_f32[blob_f16[(block_index + j) * half_type_size]]
					local block_offset = ((block_index + j) * type_size) + 2
                    
                    --[[
                    -- a little bit slower, dunno why it's not being unrolled?
                    for i = 0, 15 do
                        local byte = a[block_offset + band(i, half_block_size - 1) + 2]
                        result = result + (band(byte, 0x0F) - 8) * b[j * 32 + i] * scale
                        result = result + (band(rshift(byte, 4), 0x0F) - 8) * b[j * 32 + i + 15] * scale
                    end
                    ]]

                    local b0 = a[block_offset + band(0, half_block_size - 1)]
                    local b1 = a[block_offset + band(1, half_block_size - 1)]
                    local b2 = a[block_offset + band(2, half_block_size - 1)]
                    local b3 = a[block_offset + band(3, half_block_size - 1)]
                    local b4 = a[block_offset + band(4, half_block_size - 1)]
                    local b5 = a[block_offset + band(5, half_block_size - 1)]
                    local b6 = a[block_offset + band(6, half_block_size - 1)]
                    local b7 = a[block_offset + band(7, half_block_size - 1)]
                    local b8 = a[block_offset + band(8, half_block_size - 1)]
                    local b9 = a[block_offset + band(9, half_block_size - 1)]
                    local b10 = a[block_offset + band(10, half_block_size - 1)]
                    local b11 = a[block_offset + band(11, half_block_size - 1)]
                    local b12 = a[block_offset + band(12, half_block_size - 1)]
                    local b13 = a[block_offset + band(13, half_block_size - 1)]
                    local b14 = a[block_offset + band(14, half_block_size - 1)]
                    local b15 = a[block_offset + band(15, half_block_size - 1)]

                    j = j * 32

                    result = result + (band(b0, 0x0F) - 8) * scale * b[j + 0]
                    result = result + (band(b1, 0x0F) - 8) * scale * b[j + 1]
                    result = result + (band(b2, 0x0F) - 8) * scale * b[j + 2]
                    result = result + (band(b3, 0x0F) - 8) * scale * b[j + 3]
                    result = result + (band(b4, 0x0F) - 8) * scale * b[j + 4]
                    result = result + (band(b5, 0x0F) - 8) * scale * b[j + 5]
                    result = result + (band(b6, 0x0F) - 8) * scale * b[j + 6]
                    result = result + (band(b7, 0x0F) - 8) * scale * b[j + 7]
                    result = result + (band(b8, 0x0F) - 8) * scale * b[j + 8]
                    result = result + (band(b9, 0x0F) - 8) * scale * b[j + 9]
                    result = result + (band(b10, 0x0F) - 8) * scale * b[j + 10]
                    result = result + (band(b11, 0x0F) - 8) * scale * b[j + 11]
                    result = result + (band(b12, 0x0F) - 8) * scale * b[j + 12]
                    result = result + (band(b13, 0x0F) - 8) * scale * b[j + 13]
                    result = result + (band(b14, 0x0F) - 8) * scale * b[j + 14]
                    result = result + (band(b15, 0x0F) - 8) * scale * b[j + 15]

                    result = result + (rshift(b0, 4) - 8) * scale * b[j + 16]
                    result = result + (rshift(b1, 4) - 8) * scale * b[j + 17]
                    result = result + (rshift(b2, 4) - 8) * scale * b[j + 18]
                    result = result + (rshift(b3, 4) - 8) * scale * b[j + 19]
                    result = result + (rshift(b4, 4) - 8) * scale * b[j + 20]
                    result = result + (rshift(b5, 4) - 8) * scale * b[j + 21]
                    result = result + (rshift(b6, 4) - 8) * scale * b[j + 22]
                    result = result + (rshift(b7, 4) - 8) * scale * b[j + 23]
                    result = result + (rshift(b8, 4) - 8) * scale * b[j + 24]
                    result = result + (rshift(b9, 4) - 8) * scale * b[j + 25]
                    result = result + (rshift(b10, 4) - 8) * scale * b[j + 26]
                    result = result + (rshift(b11, 4) - 8) * scale * b[j + 27]
                    result = result + (rshift(b12, 4) - 8) * scale * b[j + 28]
                    result = result + (rshift(b13, 4) - 8) * scale * b[j + 29]
                    result = result + (rshift(b14, 4) - 8) * scale * b[j + 30]
                    result = result + (rshift(b15, 4) - 8) * scale * b[j + 31]
                end
        
                out[row] = result
            end
        end

        local function kernel_f32_f32_f32(a, b, out, dim0, dim1, offset)
            for row = offset or 0, dim0 - 1 do
                local result = 0
                local offset = row * dim1
        
                for j = 0, dim1 - 1 do
                    result = result + a[offset + j] * b[j]
                end
        
                out[row] = result
            end
        end

        return {
            MatrixVectorMultiply = function(a, b, out, dim0, dim1, offset)
                if a.blob.type == "Q4_0" and b.blob.type == "F32" and out.blob.type == "F32" then
                    kernel_q40_f32_f32(a.blob.blob, b.blob.blob, out.blob.blob, dim0, dim1, offset, a.blob.cached_f16_to_f32, a.blob.blob_f16, a.blob.half_type_size, a.blob.type_size, a.blob.half_block_size)
                elseif a.blob.type == "F32" and b.blob.type == "F32" and out.blob.type == "F32" then
                    kernel_f32_f32_f32(a.blob.blob, b.blob.blob, out.blob.blob, dim0, dim1, offset)
                else
                    error("NYI")
                end
            end
    }
    elseif backend == "pthreads" then
        local pthreads = require("compute.cpu_pthreads")
        local threaded_for = pthreads.threaded_for(function(thread_start, thread_stop, dim1, out, a, b)	
            a:MatrixVectorMultiply(b, out, thread_stop, dim1, thread_start)            
        end, {"double", "@tensor", "@tensor", "@tensor"}, pthreads.get_cpu_threads())
        
        return {
            MatrixVectorMultiply = function(a, b, out, dim0, dim1)
                threaded_for(dim0, dim1, out, a, b)
            end
        }

        --[[
            function Tensor:Dot(thisOffset, that, thatOffset, size)
                local result = 0

                for j = 0, size - 1 do
                    result = result + self:GetFloat(thisOffset + j) * that:GetFloat(thatOffset + j)
                end

                return result
            end

        	function Tensor:MatrixVectorMultiply(that, out, dim0, dim1)
                for i = 0, dim0 - 1 do
                    local result = 0
            
                    for j = 0, dim1 - 1 do
                        result = result + self:GetFloat(i * dim1 + j) * that:GetFloat(j)
                    end
            
                    out:SetFloat(i, result)
                end
            end

            
            ]]
    elseif backend == "cuda" then
        local ffi = require("ffi")
        local gpu = require("compute.gpu_cuda")
        gpu.init_with_device(0)
    
        local cache, cache1, cache2
        do
            cache = ffi.new("float[65536]")
            local function host_f16_to_f32(bits)
                local sign = 1 - bit.band(bit.rshift(bits, 15), 0x1) * 2
                local exponent = bit.band(bit.rshift(bits, 10), 0x1F)
                local mantissa = bit.band(bits, 0x3FF)
                local base = mantissa + 1024
                return sign * math.ldexp(base, exponent - 25)
            end
            for i = 0, 65536 - 1 do
                cache[i] = host_f16_to_f32(i)
            end
    
            --[[
            cache1 = ffi.new("float[256]")
            cache2 = ffi.new("float[256]")
            for i = 0, 255 do
                cache1[i] = bit.band(i, 0x0F) - 8
                cache2[i] = bit.band(bit.rshift(i, 4), 0x0F) - 8
            end
            ]]
        end
    
        local kernel_q40_f32_f32 = gpu.compile_kernel([=[
            #define BLOCK_SIZE 32
            #define HALF_BLOCK_SIZE 16
            #define TYPE_SIZE 18
            #define HALF_TYPE_SIZE 9
            __device__ float f16_to_f32_cache[65536];
            //__device__ float cache1[256];
            //__device__ float cache2[256];
    
            __device__ void decode_float_block(const unsigned char *blob, int block_index, float *f) {
                const unsigned short* blob_f16 = (const unsigned short*)blob;
                
                float scale = f16_to_f32_cache[blob_f16[block_index * HALF_TYPE_SIZE]];
                
                int block_offset = block_index * TYPE_SIZE;
                const unsigned char *block = blob + block_offset;
    
                #pragma unroll
                for (int i = 0; i < HALF_BLOCK_SIZE; i++) {
                    unsigned char b = block[(i & (HALF_BLOCK_SIZE - 1)) + 2];
    
                    f[i] = ((b & 0x0F) - 8) * scale;
                    f[i+16] = (((b / 16) & 0x0F) - 8) * scale;
                    
                    // slower than the above
                    //f[i] = cache1[b] * scale;
                    //f[i+16] = cache2[b] * scale;
    
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
            f16_to_f32_cache = {data = cache, size = ffi.sizeof(cache)},
            --cache1 = {data = cache1, size = ffi.sizeof(cache1)},
            --cache2 = {data = cache2, size = ffi.sizeof(cache2)},
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

        return {
            MatrixVectorMultiply = function(a, b, out, dim0, dim1)
                if a.blob.type == "Q4_0" and b.blob.type == "F32" and out.blob.type == "F32" then
                    run_kernel(kernel_q40_f32_f32, a.blob, b.blob, out.blob, dim0, dim1)
                elseif a.blob.type == "F32" and b.blob.type == "F32" and out.blob.type == "F32" then
                    run_kernel(kernel_f32_f32_f32, a.blob, b.blob, out.blob, dim0, dim1)
                else
                    error("NYI")
                end
            end
        }
    end
end