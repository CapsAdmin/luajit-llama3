return function(backend)
	assert(
		backend == "lua" or backend == "pthreads" or backend == "cuda",
		"backend must be lua, pthreads or cuda"
	)
	local f16_to_f32 = require("gguf").f16_to_f32

	if backend == "lua" then
		local rshift = bit.rshift
		local lshift = bit.lshift
		local band = bit.band

		local function kernel_vecmul_f32_f32_f32(a, b, out, dim0, dim1, offset)
			local a = a.blob
			local b = b.blob
			local out = out.blob

			for row = offset or 0, dim0 - 1 do
				local result = 0
				local offset = row * dim1

				for j = 0, dim1 - 1 do
					result = result + a[offset + j] * b[j]
				end

				out[row] = result
			end
		end

		local function kernel_vecmul_q40_f32_f32(a, b, out, dim0, dim1, offset)
			local blob_f16 = a.blob_f16
			
			local type_size = a.type_size
			local block_size = a.block_size

			local half_type_size = a.half_type_size
			local half_block_size = a.half_block_size - 1

			local a = a.blob
			local b = b.blob
			local out = out.blob

			for row = offset or 0, dim0 - 1 do
				local result = 0
				local block_index = (row * dim1) / block_size

				for j = 0, (dim1 / block_size) - 1 do
					local scale = f16_to_f32(blob_f16[(block_index + j) * half_type_size])
					local block_offset = ((block_index + j) * type_size) + 2

                    local b00 = a[block_offset + band(0, half_block_size)]
                    local b01 = a[block_offset + band(1, half_block_size)]
                    local b02 = a[block_offset + band(2, half_block_size)]
                    local b03 = a[block_offset + band(3, half_block_size)]
                    local b04 = a[block_offset + band(4, half_block_size)]
                    local b05 = a[block_offset + band(5, half_block_size)]
                    local b06 = a[block_offset + band(6, half_block_size)]
                    local b07 = a[block_offset + band(7, half_block_size)]
                    local b08 = a[block_offset + band(8, half_block_size)]
                    local b09 = a[block_offset + band(9, half_block_size)]
                    local b10 = a[block_offset + band(10, half_block_size)]
                    local b11 = a[block_offset + band(11, half_block_size)]
                    local b12 = a[block_offset + band(12, half_block_size)]
                    local b13 = a[block_offset + band(13, half_block_size)]
                    local b14 = a[block_offset + band(14, half_block_size)]
                    local b15 = a[block_offset + band(15, half_block_size)]
					j = j * 32
                    result = result + (band(b00, 0x0F) - 8) * scale * b[j + 0] 
                    result = result + (band(b01, 0x0F) - 8) * scale * b[j + 1] 
                    result = result + (band(b02, 0x0F) - 8) * scale * b[j + 2] 
                    result = result + (band(b03, 0x0F) - 8) * scale * b[j + 3] 
                    result = result + (band(b04, 0x0F) - 8) * scale * b[j + 4] 
                    result = result + (band(b05, 0x0F) - 8) * scale * b[j + 5] 
                    result = result + (band(b06, 0x0F) - 8) * scale * b[j + 6] 
                    result = result + (band(b07, 0x0F) - 8) * scale * b[j + 7] 
                    result = result + (band(b08, 0x0F) - 8) * scale * b[j + 8] 
                    result = result + (band(b09, 0x0F) - 8) * scale * b[j + 9] 
                    result = result + (band(b10, 0x0F) - 8) * scale * b[j + 10]
                    result = result + (band(b11, 0x0F) - 8) * scale * b[j + 11]
                    result = result + (band(b12, 0x0F) - 8) * scale * b[j + 12]
                    result = result + (band(b13, 0x0F) - 8) * scale * b[j + 13]
                    result = result + (band(b14, 0x0F) - 8) * scale * b[j + 14]
                    result = result + (band(b15, 0x0F) - 8) * scale * b[j + 15]
                    result = result + (rshift(b00, 4) - 8) * scale * b[j + 16] 
                    result = result + (rshift(b01, 4) - 8) * scale * b[j + 17] 
                    result = result + (rshift(b02, 4) - 8) * scale * b[j + 18] 
                    result = result + (rshift(b03, 4) - 8) * scale * b[j + 19] 
                    result = result + (rshift(b04, 4) - 8) * scale * b[j + 20] 
                    result = result + (rshift(b05, 4) - 8) * scale * b[j + 21] 
                    result = result + (rshift(b06, 4) - 8) * scale * b[j + 22] 
                    result = result + (rshift(b07, 4) - 8) * scale * b[j + 23] 
                    result = result + (rshift(b08, 4) - 8) * scale * b[j + 24] 
                    result = result + (rshift(b09, 4) - 8) * scale * b[j + 25] 
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
		
		return {
			MatrixVectorMultiply = function(a, b, out, dim0, dim1, offset)
				if a.blob.type == "Q4_0" and b.blob.type == "F32" and out.blob.type == "F32" then
					kernel_vecmul_q40_f32_f32(a.blob, b.blob, out.blob, dim0, dim1, offset)
				elseif a.blob.type == "F32" and b.blob.type == "F32" and out.blob.type == "F32" then
					kernel_vecmul_f32_f32_f32(a.blob, b.blob, out.blob, dim0, dim1, offset)
				else
					error("NYI " .. a.blob.type .. "*" .. out.blob.type)
				end
			end,
		}
	elseif backend == "pthreads" then
		local pthreads = require("compute.cpu_pthreads")
		local threaded_for = pthreads.threaded_for(
			function(thread_start, thread_stop, dim1, out, a, b)
				a:MatrixVectorMultiply(b, out, thread_stop, dim1, thread_start)
			end,
			{"double", "@tensor", "@tensor", "@tensor"},
			pthreads.get_cpu_threads()
		)
		return {
			MatrixVectorMultiply = function(a, b, out, dim0, dim1)
				threaded_for(dim0, dim1, out, a, b)
			end,
		}
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

		local kernel_vecmul_q40_f32_f32 = gpu.compile_kernel(
			[=[
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
        ]=],
			"kernel_q40_f32_f32",
			{
				f16_to_f32_cache = {data = cache, size = ffi.sizeof(cache)},
			--cache1 = {data = cache1, size = ffi.sizeof(cache1)},
			--cache2 = {data = cache2, size = ffi.sizeof(cache2)},
			}
		)
		local kernel_vecmul_f32_f32_f32 = gpu.compile_kernel(
			[[
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
        ]],
			"kernel_f32_f32_f32"
		)
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
			gpu.run_kernel(kernel, thread_count, 1, 1, block_count, 1, 1, args)
			gpu.copy_from_device(out.gpu_ptr, out.blob, dim0 * out.byte_stride)
		end

		return {
			MatrixVectorMultiply = function(a, b, out, dim0, dim1)
				if a.blob.type == "Q4_0" and b.blob.type == "F32" and out.blob.type == "F32" then
					run_kernel(kernel_vecmul_q40_f32_f32, a.blob, b.blob, out.blob, dim0, dim1)
				elseif a.blob.type == "F32" and b.blob.type == "F32" and out.blob.type == "F32" then
					run_kernel(kernel_vecmul_f32_f32_f32, a.blob, b.blob, out.blob, dim0, dim1)
				else
					error("NYI")
				end
			end,
		}
	end
end
