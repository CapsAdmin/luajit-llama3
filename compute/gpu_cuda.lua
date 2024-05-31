local ffi = require("ffi")
local gpu = {}

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


gpu.cuda = cuda
gpu.nvrtc = nvrtc

function gpu.source_to_ptx(source)
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

function gpu.compile_ptx(ptx, name)
    local module = ffi.new("CUmodule[1]")
    cuda.cuModuleLoadData(module, ptx)

    local kernel = ffi.new("CUfunction[1]")
    cuda.cuModuleGetFunction(kernel, module[0], name)
    assert(kernel[0] ~= nil)

    return kernel[0]
end

function gpu.create_context()
    local device = ffi.new("int[1]")
    cuda.cuDeviceGetCount(device)
    assert(device[0] > 0, "No CUDA devices found")

    cuda.cuDeviceGet(device, 0)

    local context = ffi.new("CUcontext[1]")
    cuda.cuCtxCreate(context, 0, device[0])

    return context[0]
end

function gpu.allocate(size, buffer)
    local ptr = ffi.new("void*[1]")
    cuda.cuMemAlloc(ptr, size)
    
    if buffer then
        cuda.cuMemcpyHtoD(ptr[0], buffer, size)
    end

    return ptr
end

return gpu