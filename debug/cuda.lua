local ffi = require("ffi")

ffi.cdef[[
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef void *nvrtcProgram;
typedef int CUdevice;
typedef int nvrtcResult;

const char* cudaGetErrorString(int error);
int cuGetErrorString(int error, const char **str);


const char* nvrtcGetErrorString(nvrtcResult result);
nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name,
                               int numHeaders, const char** headers, const char** includeNames);
nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog);
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options);
nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet);
nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx);
nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet);
nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log);

int cudaMalloc(void **devPtr, size_t size);
int cudaMemcpy(void *dst, const void *src, size_t count, int kind);
int cudaFree(void *devPtr);

int cuInit(unsigned int Flags);
int cuDeviceGetCount(int *count);
int cuDeviceGet(CUdevice *device, int ordinal);
int cuCtxSetCurrent ( CUcontext ctx );
int cuCtxDestroy ( CUcontext ctx );
int cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
int cuModuleLoadData(CUmodule *module, const void *image);
int cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
int cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                   unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
int cuPeekAtLastError ( void );
int cuMemAlloc(void **dptr, size_t bytesize);
int cuMemcpyHtoD(void *dstDevice, const void *srcHost, size_t ByteCount);
int cuMemcpyDtoH(void *dstHost, const void *srcDevice, size_t ByteCount);
int cuMemFree(void *dptr);
]]
local cudaMemcpyHostToDevice = 1
local cudaMemcpyDeviceToHost = 2

local libnvrtc = ffi.load("nvrtc")
local libcuda = ffi.load("cuda")

local cuda = setmetatable({}, {
    __index = function(s, key)
        local f = libcuda[key]
        return function(...) 
            local res = f(...)
            if res ~= 0 then
                local str = ffi.new("const char*[512]")
                libcuda.cuGetErrorString(res, ffi.cast("const char **", str))
                error(key .. " failed with code ("..res.."): " .. ffi.string(str[0]), 2)
            end
        end
    end
})

local nvrtc = setmetatable({}, {
    __index = function(s, key)
        local f = libnvrtc[key]
        return function(...) 
            local res = f(...)
            if res ~= 0 then
                local prog = assert(_G.PROG)
                local logSize = ffi.new("size_t[1]")
                libnvrtc.nvrtcGetProgramLogSize(prog[0], logSize)
                local log = ffi.new("char[?]", logSize[0])
                libnvrtc.nvrtcGetProgramLog(prog[0], log)
                log = ffi.string(log)

                error(key .. " failed with code ("..res.."): " .. ffi.string(libnvrtc.nvrtcGetErrorString(res)) .. "\n" .. log, 2)
            end
        end
    end
})


local function source_to_ptx(source)
    local prog = ffi.new("nvrtcProgram[1]")
    nvrtc.nvrtcCreateProgram(prog, source, nil, 0, nil, nil)
    _G.PROG = prog -- for debug
    res = nvrtc.nvrtcCompileProgram(prog[0], 0, nil)

    local ptxSize = ffi.new("size_t[1]")
    nvrtc.nvrtcGetPTXSize(prog[0], ptxSize)
    local ptx = ffi.new("char[?]", ptxSize[0])
    nvrtc.nvrtcGetPTX(prog[0], ptx)
    nvrtc.nvrtcDestroyProgram(prog)

    return ffi.string(ptx, ptxSize[0])
end

local function compile_ptx(ptx, name)
    local module = ffi.new("CUmodule[1]")
    cuda.cuModuleLoadData(module, ptx)

    local kernel = ffi.new("CUfunction[1]")
    cuda.cuModuleGetFunction(kernel, module[0], name)
    assert(kernel[0] ~= nil)

    return kernel[0]
end

local function create_context()
    local device = ffi.new("int[1]")
    cuda.cuDeviceGetCount(device)
    assert(device[0] > 0, "No CUDA devices found")

    cuda.cuDeviceGet(device, 0)

    local context = ffi.new("CUcontext[1]")
    cuda.cuCtxCreate(context, 0, device[0])

    return context[0]
end

local function test_kernel()
    cuda.cuInit(0)

    local context = create_context()
    cuda.cuCtxSetCurrent(context)

    local kernel = compile_ptx(source_to_ptx([[
        extern "C" __global__ void test() {
            // Does nothing
        }
    ]]), "test")

    local args = ffi.new("void*[1]", {})
    cuda.cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nil, args, nil)

    cuda.cuCtxDestroy(context)
end

test_kernel()

local function gpu_allocate(size, buffer)
    local ptr = ffi.new("void*[1]")
    cuda.cuMemAlloc(ptr, size)
    
    if buffer then
        cuda.cuMemcpyHtoD(ptr[0], buffer, size)
    end

    return ptr
end

local function runKernel(ptx, name, a, b, out, sx, sy, numBlocks, threadsPerBlock)
    cuda.cuInit(0)

    local context = create_context()
    cuda.cuCtxSetCurrent(context)

    local kernel = compile_ptx(ptx, name)

    local F32_SIZE = 4

    local device_a = gpu_allocate(a.size*F32_SIZE, a.blob.blob)
    local device_b = gpu_allocate(b.size*F32_SIZE, b.blob.blob)

    local device_out = gpu_allocate(out.size*F32_SIZE)

    local box_sx = ffi.new("int[1]", sx)
    local box_sy = ffi.new("int[1]", sy)
    local args = ffi.new("void*[5]", device_a, device_b, device_out, box_sx, box_sy)
    cuda.cuLaunchKernel(
        kernel, 
        threadsPerBlock, 1, 1, 
        numBlocks, 1, 1, 

        0, nil, args, nil
    )

    cuda.cuMemcpyDtoH(out.blob.blob, device_out[0], out.size*F32_SIZE)

    cuda.cuMemFree(device_a[0])
    cuda.cuMemFree(device_b[0])
    cuda.cuMemFree(device_out[0])

    cuda.cuCtxDestroy(context)
end


package.path = './?.lua;' .. package.path
local Tensor = require("tensor")

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
    runKernel(source_to_ptx([[
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

if false then
    local measure = require("debug.measure")

    local total = 0

    for i = 1, 5 do
        if i > 2 then -- jit warmup
        measure("MatrixVectorMultiply") end

        for k, v in ipairs(variants) do
            v.a:MatrixVectorMultiply(v.b, v.out, v.dim0, v.dim1)
        end

        if i > 2 then total = total + measure() end
    end

    print("avg time: ", total / 20)
end