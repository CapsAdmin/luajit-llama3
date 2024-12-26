local ffi = require("ffi")
ffi.cdef[[
	void *malloc( size_t size );
	void *memcpy(void *dest, const void *src, size_t n);
]]
local math_exp = math.exp
local Tensor = {}
Tensor.__index = Tensor
Tensor.tensors_created = {}

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

	function Tensor:MatrixVectorMultiply(that, out, dim0, dim1, offset)
		for i = offset or 0, dim0 - 1 do
			local result = 0

			for j = 0, dim1 - 1 do
				result = result + self:GetFloat(i * dim1 + j) * that:GetFloat(j)
			end

			out:SetFloat(i, result)
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
	local type_map = {
		F32 = 0,
		Q4_0 = 1,
		Q8_0 = 2,
		Q6_K = 3,
	}

	-- double lookup
	for i, str in pairs(type_map) do
		type_map[str] = i
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

function Tensor:UseComputeKernel(backend)
	local functions

	if backend == "lua" then
		local ggf = require("gguf")
		local f16_to_f32 = ggf.f16_to_f32
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

		local function kernel_vecmul_q6k_f32_f32(a, b, out, dim0, dim1, offset)
			local blob_f16 = a.blob_f16
			local type_size = a.type_size
			local block_size = a.block_size
			local a = a.blob
			local b = b.blob
			local out = out.blob

			for row = offset or 0, dim0 - 1 do
				local result = 0
				local block_index = (row * dim1) / block_size

				for j = 0, (dim1 / block_size) - 1 do
					local d = f16_to_f32(blob_f16[(block_index + j) * 2])
					local block_offset = ((block_index + j) * type_size) + 2

					-- Match C code layout: process 128 values at a time
					for n = 0, 128 - 1, 128 do
						for l = 0, 31 do
							-- Get indices for ql, qh sections
							local is = rshift(l, 4) -- l/16
							-- First value: low 4 bits of ql[l] | 2 bits from qh[l] shifted
							local q1 = band(a[block_offset + l], 0xF)
							q1 = q1 + lshift(band(rshift(a[block_offset + 64 + l], 0), 3), 4)
							q1 = q1 - 32
							-- Second value: low 4 bits of ql[l+32] | 2 bits from qh[l] shifted
							local q2 = band(a[block_offset + l + 32], 0xF)
							q2 = q2 + lshift(band(rshift(a[block_offset + 64 + l], 2), 3), 4)
							q2 = q2 - 32
							-- Third value: high 4 bits of ql[l] | 2 bits from qh[l] shifted
							local q3 = rshift(a[block_offset + l], 4)
							q3 = q3 + lshift(band(rshift(a[block_offset + 64 + l], 4), 3), 4)
							q3 = q3 - 32
							-- Fourth value: high 4 bits of ql[l+32] | 2 bits from qh[l] shifted
							local q4 = rshift(a[block_offset + l + 32], 4)
							q4 = q4 + lshift(band(rshift(a[block_offset + 64 + l], 6), 3), 4)
							q4 = q4 - 32
							-- Accumulate results with correct indexing
							local b_idx = j * block_size
							result = result + d * q1 * b[b_idx + l]
							result = result + d * q2 * b[b_idx + l + 32]
							result = result + d * q3 * b[b_idx + l + 64]
							result = result + d * q4 * b[b_idx + l + 96]
						end
					end
				end

				out[row] = result
			end
		end

		functions = {
			MatrixVectorMultiply = function(a, b, out, dim0, dim1, offset)
				if a.type == "Q4_0" and b.type == "F32" and out.type == "F32" then
					kernel_vecmul_q40_f32_f32(a, b, out, dim0, dim1, offset)
				elseif a.type == "F32" and b.type == "F32" and out.type == "F32" then
					kernel_vecmul_f32_f32_f32(a, b, out, dim0, dim1, offset)
				elseif a.type == "Q6_K" and b.type == "F32" and out.type == "F32" then
					kernel_vecmul_q6k_f32_f32(a, b, out, dim0, dim1, offset)
				else
					error("NYI " .. a.type .. "*" .. out.type)
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
		functions = {
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

		functions = {
			MatrixVectorMultiply = function(a, b, out, dim0, dim1)
				if a.type == "Q4_0" and b.type == "F32" and out.type == "F32" then
					run_kernel(kernel_vecmul_q40_f32_f32, a, b, out, dim0, dim1)
				elseif a.type == "F32" and b.type == "F32" and out.type == "F32" then
					run_kernel(kernel_vecmul_f32_f32_f32, a, b, out, dim0, dim1)
				else
					error("NYI")
				end
			end,
		}
	else
		error("backend must be lua, pthreads or cuda, got " .. tostring(backend))
	end

	for k, v in pairs(functions) do
		assert(Tensor[k], k .. " is not a function")
		Tensor[k] = v
	end

	return Tensor
end

local tensor_types = {}

do
	local ggf = require("gguf")
	local f16_to_f32 = ggf.f16_to_f32
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
			MatrixVectorMultiply = function(a, b, out, dim0, dim1, offset)
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

	do
		local block_size = ggf.GGMLTypeMap.Q4_0.block_size
		local half_block_size = block_size / 2
		local type_size = ggf.GGMLTypeMap.Q4_0.type_size
		local half_type_size = type_size / 2
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
				blob_f16 = blob_f16,
				half_type_size = half_type_size,
				type_size = type_size,
				half_block_size = half_block_size,
				block_size = block_size,
				GetFloat = function(_, index)
					local block_index = rshift(index, 5)
					local block_offset = block_index * type_size
					local scale = f16_to_f32(blob_f16[block_index * half_type_size])
					local modIndex = band(index, block_size - 1)
					local base_offset = block_offset + band(modIndex, half_block_size - 1)
					local shift_amount = rshift(modIndex, 4) * 4
					local quant = band(rshift(blob[2 + base_offset], shift_amount), 0x0F)
					return (quant - 8) * scale
				end,
				MatrixVectorMultiply2 = function(a, b, out, dim0, dim1, offset)
					assert(b.type == "F32")
					assert(out.type == "F32")
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
				end,
			}
		end
	end

	do
		local block_size = ggf.GGMLTypeMap.Q6_K.block_size
		local half_block_size = block_size / 2
		local type_size = ggf.GGMLTypeMap.Q6_K.type_size
		local half_type_size = type_size / 2
		local rshift = bit.rshift
		local band = bit.band
		local lshift = bit.lshift
		tensor_types.Q6_K = function(size, blob)
			local byte_size = size * type_size
			blob = ffi.cast("uint8_t*", blob or ffi.C.malloc(byte_size))
			local blob_f16 = ffi.cast("uint16_t*", blob)
			assert(byte_size % type_size == 0, "Total size must be a multiple of the block size")
			byte_size = byte_size / type_size
			local floats = ffi.typeof("float[256]") -- Increased to 256 for Q6_K block size
			local f = floats()
			return {
				blob = blob,
				size = tonumber(size),
				byte_size = tonumber(byte_size),
				byte_stride = 1,
				blob_f16 = blob_f16,
				half_type_size = half_type_size,
				block_size = block_size,
				half_block_size = half_block_size,
				type_size = type_size,
				GetFloat = function(_, index)
					local block_index = rshift(index, 8) -- Divide by 256
					local block_offset = block_index * type_size
					local scale = f16_to_f32(blob_f16[block_index * 2])
					local min = f16_to_f32(blob_f16[block_index * 2 + 1])
					-- Calculate position within block
					local modIndex = band(index, 255) -- index % 256
					-- Q6_K uses 6 bits per value
					-- Each byte contains 1.333 values (8/6 bits)
					-- Need to handle bit extraction carefully
					local byte_idx = rshift(modIndex * 6, 3) -- (modIndex * 6) / 8
					local bit_shift = band(modIndex * 6, 7) -- (modIndex * 6) % 8
					-- Extract 6 bits across byte boundary if needed
					local val = band(rshift(blob[block_offset + 4 + byte_idx], bit_shift), 0x3F)

					if bit_shift > 2 then
						-- Need some bits from next byte
						val = band(val + lshift(blob[block_offset + 5 + byte_idx], 8 - bit_shift), 0x3F)
					end

					-- Dequantize: val * scale + min
					return val * scale + min
				end,
				MatrixVectorMultiply = function(a, b, out, dim0, dim1, offset)
					local blob_f16 = a.blob_f16
					local type_size = a.type_size
					local block_size = a.block_size
					local a = a.blob
					local b = b.blob
					local out = out.blob

					for row = offset or 0, dim0 - 1 do
						local result = 0
						local block_index = (row * dim1) / block_size

						for j = 0, (dim1 / block_size) - 1 do
							local d = f16_to_f32(blob_f16[(block_index + j) * 2])
							local block_offset = ((block_index + j) * type_size) + 2

							-- Match C code layout: process 128 values at a time
							for n = 0, 128 - 1, 128 do
								for l = 0, 31 do
									-- Get indices for ql, qh sections
									local is = rshift(l, 4) -- l/16
									-- First value: low 4 bits of ql[l] | 2 bits from qh[l] shifted
									local q1 = band(a[block_offset + l], 0xF)
									q1 = q1 + lshift(band(rshift(a[block_offset + 64 + l], 0), 3), 4)
									q1 = q1 - 32
									-- Second value: low 4 bits of ql[l+32] | 2 bits from qh[l] shifted
									local q2 = band(a[block_offset + l + 32], 0xF)
									q2 = q2 + lshift(band(rshift(a[block_offset + 64 + l], 2), 3), 4)
									q2 = q2 - 32
									-- Third value: high 4 bits of ql[l] | 2 bits from qh[l] shifted
									local q3 = rshift(a[block_offset + l], 4)
									q3 = q3 + lshift(band(rshift(a[block_offset + 64 + l], 4), 3), 4)
									q3 = q3 - 32
									-- Fourth value: high 4 bits of ql[l+32] | 2 bits from qh[l] shifted
									local q4 = rshift(a[block_offset + l + 32], 4)
									q4 = q4 + lshift(band(rshift(a[block_offset + 64 + l], 6), 3), 4)
									q4 = q4 - 32
									-- Accumulate results with correct indexing
									local b_idx = j * block_size
									result = result + d * q1 * b[b_idx + l]
									result = result + d * q2 * b[b_idx + l + 32]
									result = result + d * q3 * b[b_idx + l + 64]
									result = result + d * q4 * b[b_idx + l + 96]
								end
							end
						end

						out[row] = result
					end
				end,
			}
		end
	end

	do
		local block_size = 32 -- Standard block size
		local type_size = block_size -- For Q8_0, type_size equals block_size since each value is 1 byte
		local rshift = bit.rshift
		local band = bit.band
		tensor_types.Q8_0 = function(size, blob)
			local byte_size = size
			blob = ffi.cast("int8_t*", blob or ffi.C.malloc(byte_size))
			assert(byte_size % block_size == 0, "Total size must be a multiple of the block size")
			byte_size = byte_size / block_size
			local floats = ffi.typeof("float[32]")
			local f = floats()
			return {
				blob = blob,
				size = tonumber(size),
				byte_size = tonumber(byte_size),
				byte_stride = 1,
				-- Get a single float value at the given index
				GetFloat = function(_, index)
					-- Q8_0 values are direct 8-bit integers, no scaling needed
					return tonumber(blob[index])
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

return Tensor
