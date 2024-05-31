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
    local gpu = require("gpu_cuda")

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

        extern "C" __global__ void test(const unsigned char *self, float* that, float* out, int dim0, int dim1) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < dim0) { 
                float result = 0.0f;
                for (int j = 0; j < dim1; j++) {
                    result += get_float(self, row * dim1 + j) * that[j];
                }
                out[row] = result;
            }
        }
    ]]), "test")


    do
        local F32_SIZE = 4
        local SHORT_SIZE = 2

        Tensor.MatrixVectorMultiplyCPU = Tensor.MatrixVectorMultiply

        local f32cache = {}
        local q4cache = {}
        local f322cache = {}

        function Tensor:MatrixVectorMultiply(that, out, dim0, dim1)
            if self.blob.type == "Q4_0" and that.blob.type == "F32" and out.blob.type == "F32" then
                self:MatrixVectorMultiplyCPU(that, out, dim0, dim1)

                local device_a = q4cache[self.size]
                if not device_a then
                    device_a = gpu.allocate(self.size*SHORT_SIZE)	
                    q4cache[self.size] = device_a
                    print("q4", self.size)
                end

                local device_b = f32cache[that.size]
                if not device_b then
                    device_b = gpu.allocate(that.size*F32_SIZE)	
                    f32cache[that.size] = device_b
                    print("f32", that.size)
                end

                local device_out = f322cache[out.size]
                if not device_out then
                    device_out = gpu.allocate(out.size*F32_SIZE)	
                    f322cache[out.size] = device_out
                    print("f322", self.size)
                end

                gpu.cuda.cuMemcpyHtoD(device_a[0], self.blob.blob, self.size*SHORT_SIZE)
                gpu.cuda.cuMemcpyHtoD(device_b[0], that.blob.blob, that.size*F32_SIZE)
                                    
                local threadsPerBlock = 64
                local numBlocks = math.ceil(dim0 / threadsPerBlock)

                local box_sx = ffi.new("int[1]", dim0)
                local box_sy = ffi.new("int[1]", dim1)
                local args = ffi.new("void*[5]", device_a, device_b, device_out, box_sx, box_sy)
                gpu.cuda.cuLaunchKernel(
                    kernel_f32_q4_0_f32, 
                    threadsPerBlock, 1, 1, 
                    numBlocks, 1, 1, 
                    0, nil, args, nil
                )

                local expected = out
                local out = Tensor:F32(out.size)

                gpu.cuda.cuMemcpyDtoH(out.blob.blob, device_out[0], out.size*F32_SIZE)

                for i = 0, out.size - 1 do
                    local a, b = out:GetFloat(i), expected:GetFloat(i)
            
                    if (a - b) > 0.01 then
                        print(a, " ~= ", b, (a - b))
                        error("")
                        break
                    end
                end
                 
                print("success")
            else
                self:MatrixVectorMultiplyCPU(that, out, dim0, dim1)
            end
        end
    end
end

return {
    use_cuda = use_cuda,
    use_pthreads = use_pthreads,
}