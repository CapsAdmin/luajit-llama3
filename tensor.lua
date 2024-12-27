local ffi = require("ffi")
ffi.cdef[[
	void *malloc( size_t size );
	void *memcpy(void *dest, const void *src, size_t n);
]]
local ggf = require("gguf")
local f16_to_f32 = ggf.f16_to_f32
local has_gpu, gpu = pcall(require, "compute.gpu_cuda")
local has_cpu_threads, threads = pcall(require, "compute.cpu_threads")

if has_gpu and not IS_THREAD then
	if not gpu.initialized then
		gpu.init_with_device(0)
		gpu.initialized = true
	end
end

local math_exp = math.exp
local Tensor = {}
Tensor.__index = Tensor
Tensor.tensors_created = {}
Tensor.backend = "cpu_threads"

function Tensor.GetAll()
	return Tensor.tensors_created
end

function Tensor:SetName(n)
	self.name = n
	return self
end

function Tensor:__tostring()
	if self.name then return self.name .. "[" .. tostring(self.size) .. "]" end

	return "Tensor[" .. self.size .. "]"
end

do
	function Tensor:Dot(thisOffset, that, thatOffset, size)
		local result = 0

		for j = 0, size - 1 do
			result = result + self:GetFloat(thisOffset + j) * that:GetFloat(thatOffset + j)
		end

		return result
	end

	function Tensor:MatrixVectorMultiplyWithOffset(that, out, dim0, dim1, offset)
		error("this shouldn't run")

		for i = offset or 0, dim0 - 1 do
			local result = 0

			for j = 0, dim1 - 1 do
				result = result + self:GetFloat(i * dim1 + j) * that:GetFloat(j)
			end

			out:SetFloat(i, result)
		end
	end

	do
		function Tensor:MatrixVectorMultiplyWithOffsetCPUThreads(that, out, dim0, dim1, offset)
			if has_cpu_threads and not threaded_for then
				threaded_for = threads.threaded_for(
					function(thread_start, thread_stop, dim1, out, a, b)
						a:MatrixVectorMultiplyWithOffset(b, out, thread_stop, dim1, thread_start)
					end,
					{"double", "@tensor", "@tensor", "@tensor"},
					threads.get_cpu_threads()
				)
			end

			threaded_for(dim0, dim1, out, self, that)
		end
	end

	function Tensor:MatrixVectorMultiply(that, out, dim0, dim1, offset)
		if self.backend == "lua" then
			self:MatrixVectorMultiplyWithOffset(that, out, dim0, dim1, offset)
		elseif self.backend == "gpu" then
			self:MatrixVectorMultiplyWithOffsetGPU(that, out, dim0, dim1, offset)
		elseif self.backend == "cpu_threads" then
			self:MatrixVectorMultiplyWithOffsetCPUThreads(that, out, dim0, dim1, offset)
		else
			error("invalid backend: " .. slef.backend)
		end
	end
end

do -- avoid using these except for when debugging
	function Tensor:Reduce(thisOffset, size, seed, reduce_callback)
		local result = seed

		for i = 0, size - 1 do
			result = reduce_callback(result, self:GetFloat(thisOffset + i))
		end

		return result
	end

	function Tensor:MapInPlace(thisOffset, size, F, a, b, c, d)
		local endOffset = thisOffset + size

		for i = thisOffset, endOffset - 1 do
			self:SetFloat(i, F(self:GetFloat(i), i, a, b, c, d))
		end

		return self
	end
end

do
	function Tensor:Sum(thisOffset, size)
		local res = 0

		for i = 0, size - 1 do
			res = res + self:GetFloat(thisOffset + i)
		end

		return res
	end

	local max = math.max

	function Tensor:Max(thisOffset, size)
		local res = 0

		for i = 0, size - 1 do
			res = max(res, self:GetFloat(thisOffset + i))
		end

		return res
	end
end

function Tensor:CopyTo(thisOffset, that, thatOffset, size)
	if self.type == "F32" and that.type == "F32" then
		ffi.C.memcpy(that.blob + thatOffset, self.blob + thisOffset, size * self.byte_stride)
	else
		for i = thatOffset, thatOffset + size - 1 do
			that:SetFloat(i, self:GetFloat(i - thatOffset + thisOffset))
		end
	end
end

function Tensor:FillInPlace(thisOffset, size, identity)
	error("NYI", 2)
end

do
	function Tensor:DivideInPlace(thisOffset, size, value)
		for i = thisOffset, thisOffset + size - 1 do
			self:SetFloat(i, self:GetFloat(i) / value)
		end
	end

	function Tensor:AddTensorInPlaceOffset(thisOffset, that, thatOffset, size)
		for i = thisOffset, thisOffset + size - 1 do
			self:SetFloat(i, self:GetFloat(i) + that:GetFloat(i - thisOffset + thatOffset))
		end
	end

	function Tensor:AddTensorInPlace(that)
		for i = 0, self.size - 1 do
			self:SetFloat(i, self:GetFloat(i) + that:GetFloat(i))
		end
	end

	function Tensor:MultiplyTensorInPlaceOffset(thisOffset, that, thatOffset, size)
		for i = thisOffset, thisOffset + size - 1 do
			self:SetFloat(i, self:GetFloat(i) * that:GetFloat(i - thisOffset + thatOffset))
		end
	end

	function Tensor:MultiplyTensorInPlace(that)
		self:MultiplyTensorInPlaceOffset(0, that, 0, self.size)
	end

	function Tensor:SoftMaxInPlace(thisOffset, size)
		local max_value = self:Max(thisOffset, size)

		for i = thisOffset, thisOffset + size - 1 do
			self:SetFloat(i, math_exp(self:GetFloat(i) - max_value))
		end

		self:DivideInPlace(thisOffset, size, self:Sum(thisOffset, size))
	end
end

function Tensor:SaxpyInPlace(thisOffset, that, thatOffset, size, a)
	for i = 0, size - 1 do
		self:SetFloat(thisOffset + i, a * that:GetFloat(thatOffset + i) + self:GetFloat(thisOffset + i))
	end
end

function Tensor:SigmoidInPlace()
	for i = 0, self.size - 1 do
		local value = self:GetFloat(i)
		self:SetFloat(i, value / (1.0 + math_exp(-value)))
	end
end

function Tensor:RmsNormInPlace(x, weight, size, rmsNormEps)
	local ss = 0

	for i = 0, size - 1 do
		local f = x:GetFloat(i)
		ss = ss + f * f
	end

	ss = ss / size
	ss = ss + rmsNormEps
	ss = 1.0 / math.sqrt(ss)

	for i = 0, size - 1 do
		self:SetFloat(i, weight:GetFloat(i) * (ss * x:GetFloat(i)))
	end
end

function Tensor:UseComputeKernel(backend)
	assert(
		backend == "gpu" or backend == "cpu_threads" or backend == "lua",
		"backend must be gpu, cpu_threads or lua"
	)
	Tensor.backend = backend

	if backend == "gpu" then
		if not has_gpu then error("gpu not available") end
	elseif backend == "cpu_threads" then
		if not has_cpu_threads then
			error("cpu threads are not available: " .. threads)
		end
	end

	return Tensor
end

local tensor_types = {}

do
	do -- F64
		tensor_types.F64 = function(size, blob)
			local stride = ffi.sizeof("double")
			blob = ffi.cast("double*", blob or ffi.cast("double*", ffi.C.malloc(size * stride)))
			return {
				blob = blob,
				size = tonumber(size),
				byte_size = tonumber(size * stride),
				byte_stride = stride,
				SetFloat = function(_, index, val)
					blob[index] = val
				end,
				GetFloat = function(_, index)
					return blob[index]
				end,
			}
		end
	end

	do -- F32
		local kernel_vecmul_f32_f32_f32 = has_gpu and
			gpu.compile_kernel(
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

		local function run_kernel(kernel, a, b, out, dim0, dim1)
			-- this assumes a, b and out have been uploaded and allocated on the gpu
			-- it also assumes a never changes, which in the context of this proejct are the model weights
			gpu.copy_to_device(b.gpu_ptr, b.blob, b.byte_size)
			local thread_count = 1024
			local block_count = math.ceil((dim0 + thread_count - 1) / thread_count)
			local box_dim0 = ffi.new("int[1]", dim0)
			local box_dim1 = ffi.new("int[1]", dim1)
			local args = ffi.new("void*[5]", a.gpu_ptr, b.gpu_ptr, out.gpu_ptr, box_dim0, box_dim1)
			gpu.run_kernel(kernel, thread_count, 1, 1, block_count, 1, 1, args)
			gpu.copy_from_device(out.gpu_ptr, out.blob, dim0 * out.byte_stride)
		end

		tensor_types.F32 = function(size, blob)
			local stride = 4
			blob = ffi.cast("float*", blob or ffi.cast("float*", ffi.C.malloc(size * stride)))
			return {
				blob = blob,
				size = tonumber(size),
				byte_size = tonumber(size * stride),
				byte_stride = stride,
				SetFloat = function(_, index, val)
					blob[index] = val
				end,
				GetFloat = function(_, index)
					return blob[index]
				end,
				FillInPlace = function(self, thisOffset, size, identity)
					if identity == 0 then
						ffi.fill(self.blob + thisOffset, size * self.byte_stride, 0)
					else
						for i = thisOffset, thisOffset + size - 1 do
							self:SetFloat(i, identity)
						end
					end
				end,
				MatrixVectorMultiplyWithOffsetGPU = function(a, b, out, dim0, dim1, offset)
					run_kernel(kernel_vecmul_f32_f32_f32, a, b, out, dim0, dim1)
				end,
				MatrixVectorMultiplyWithOffset = function(a, b, out, dim0, dim1, offset)
					assert(b.type == "F32")
					assert(out.type == "F32")
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
				end,
			}
		end
	end

	do -- Q4_0
		local block_size = ggf.GGMLTypeMap.Q4_0.block_size
		local half_block_size = (block_size / 2) - 1
		local type_size = ggf.GGMLTypeMap.Q4_0.type_size
		local half_type_size = type_size / 2
		local kernel_vecmul_q40_f32_f32 = has_gpu and
			gpu.compile_kernel(
				[=[
            #define BLOCK_SIZE ]=] .. block_size .. [=[ 
            #define HALF_BLOCK_SIZE ]=] .. half_block_size .. [=[ 
            #define TYPE_SIZE ]=] .. type_size .. [=[ 
            #define HALF_TYPE_SIZE ]=] .. half_type_size .. [=[ 
            __device__ float f16_to_f32_cache[]=] .. ffi.sizeof(ggf.f16_to_f32_cache) .. [=[ ];
    
            __device__ void decode_float_block(const unsigned char *blob, int block_index, float *f) {
                const unsigned short* blob_f16 = (const unsigned short*)blob;
                
                float scale = f16_to_f32_cache[blob_f16[block_index * HALF_TYPE_SIZE]];
                
                int block_offset = block_index * TYPE_SIZE;
                const unsigned char *block = blob + block_offset;
    
                #pragma unroll
                for (int i = 0; i <= HALF_BLOCK_SIZE; i++) {
                    unsigned char b = block[(i & (HALF_BLOCK_SIZE)) + 2];
    
                    f[i] = ((b & 0x0F) - 8) * scale;
                    f[i+16] = (((b / 16) & 0x0F) - 8) * scale;
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
					f16_to_f32_cache = {data = ggf.f16_to_f32_cache, size = ffi.sizeof(ggf.f16_to_f32_cache)},
				}
			)
		local rshift = bit.rshift
		local band = bit.band
		tensor_types.Q4_0 = function(size, blob)
			local byte_size = size * type_size
			blob = ffi.cast("uint8_t*", blob or ffi.cast("uint8_t*", ffi.C.malloc(byte_size)))
			local blob_f16 = ffi.cast("uint16_t*", blob)
			assert(byte_size % block_size == 0, "Total size must be a multiple of the block size")
			byte_size = byte_size / block_size
			return {
				blob = blob,
				size = tonumber(size),
				byte_size = tonumber(byte_size),
				byte_stride = 1,
				GetFloat = function(_, index)
					local block_index = rshift(index, 5)
					local block_offset = block_index * type_size
					local scale = f16_to_f32(blob_f16[block_index * half_type_size])
					local modIndex = band(index, block_size - 1)
					local base_offset = block_offset + band(modIndex, half_block_size)
					local shift_amount = rshift(modIndex, 4) * 4
					local quant = band(rshift(blob[2 + base_offset], shift_amount), 0x0F)
					return (quant - 8) * scale
				end,
				MatrixVectorMultiplyWithOffsetGPU = function(a, b, out, dim0, dim1, offset)
					-- this assumes a, b and out have been uploaded and allocated on the gpu
					-- it also assumes a never changes, which in the context of this proejct are the model weights
					gpu.copy_to_device(b.gpu_ptr, b.blob, b.byte_size)
					local thread_count = 1024
					local block_count = math.ceil((dim0 + thread_count - 1) / thread_count)
					local box_dim0 = ffi.new("int[1]", dim0)
					local box_dim1 = ffi.new("int[1]", dim1)
					local args = ffi.new("void*[5]", a.gpu_ptr, b.gpu_ptr, out.gpu_ptr, box_dim0, box_dim1)
					gpu.run_kernel(kernel_vecmul_q40_f32_f32, thread_count, 1, 1, block_count, 1, 1, args)
					gpu.copy_from_device(out.gpu_ptr, out.blob, dim0 * out.byte_stride)
				end,
				MatrixVectorMultiplyWithOffset = function(a, b, out, dim0, dim1, offset)
					local a = blob
					assert(b.type == "F32")
					assert(out.type == "F32")
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
				end,
			}
		end
	end
end

function Tensor.New(typ, size, blob)
	if not tensor_types[typ] then error("NYI tensor type: " .. tostring(typ), 2) end

	local t = setmetatable(tensor_types[typ](size, blob), Tensor)
	t.type = typ
	table.insert(Tensor.tensors_created, t)
	return t
end

do
	local ctype = ffi.typeof([[
		struct {
			int size;
			int type;
			void *blob; 
		}
	]])
	local ctype_ptr = ffi.typeof("$*", ctype)
	local ctype_box = ffi.typeof("$[1]", ctype)
	local type_map = {}

	do
		local sorted = {}

		for k, v in pairs(tensor_types) do
			table.insert(sorted, k)
		end

		table.sort(sorted)

		for i, v in pairs(sorted) do
			type_map[v] = i
			type_map[i] = v -- double lookup
		end
	end

	function Tensor:ThreadSerialize()
		return ctype(self.size, type_map[self.type], ffi.cast("void *", self.blob))
	end

	function Tensor:ThreadDeserialize(ptr)
		local data = ffi.cast(ctype_ptr, ptr)

		if not type_map[data.type] then error("unknown type " .. data.type) end

		return Tensor.New(type_map[data.type], data.size, data.blob)
	end
end

return Tensor
