local ffi = require("ffi")
local gpu = {}

local cuda
local nvrtc 
local cublas

do
    ffi.cdef[[
        typedef struct CUctx_st *CUcontext;
        typedef struct CUmod_st *CUmodule;
        typedef struct CUfunc_st *CUfunction;
        typedef struct CUstream_st *CUstream;
        typedef int CUdevice;

        const char* cudaGetErrorString(int error);
        int cuGetErrorString(int error, const char **str);

        int cudaMalloc(void **devPtr, size_t size);
        int cudaMemcpy(void *dst, const void *src, size_t count, int kind);
        int cudaFree(void *devPtr);

        int cuInit(unsigned int Flags);
        int cuDeviceGetCount(int *count);
        int cuDeviceGet(CUdevice *device, int ordinal);
        int cuCtxSetCurrent ( CUcontext ctx );
        int cuCtxDestroy ( CUcontext ctx );
        int cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
        int cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
        
        int cuModuleLoadData(CUmodule *module, const void *image);
        int cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
        int cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
        int cuMemAlloc(void **dptr, size_t bytesize);
        int cuMemcpyHtoD(void *dstDevice, const void *srcHost, size_t ByteCount);
        int cuMemcpyDtoH(void *dstHost, const void *srcDevice, size_t ByteCount);
        int cuMemFree(void *dptr);
        int cuMemGetInfo(size_t*free, size_t*total);
    ]]

    local cudaMemcpyHostToDevice = 1
    local cudaMemcpyDeviceToHost = 2

    local libcuda = ffi.load("cuda")

    cuda = setmetatable({}, {
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

    function gpu.get_memory()
        local free = ffi.new("size_t[1]")
        local available = ffi.new("size_t[1]")
        cuda.cuMemGetInfo(free, available)
        return tonumber(free[0]), tonumber(available[0])
    end

    function gpu.init(retain)
        cuda.cuInit(0)

        local device_count = ffi.new("int[1]")
        cuda.cuDeviceGetCount(device_count)
        assert(device_count[0] > 0, "No CUDA devices found")

        local device = ffi.new("int[0]")
        cuda.cuDeviceGet(device, 0)

        local context = ffi.new("CUcontext[1]")
        if retain then
            cuda.cuDevicePrimaryCtxRetain(context, device[0])
        else
            cuda.cuCtxCreate(context, 0, device[0])
        end
        
        cuda.cuCtxSetCurrent(context[0])
    end

    function gpu.copy_to_device(device_ptr, buffer, byte_size)
        cuda.cuMemcpyHtoD(device_ptr[0], buffer, byte_size)
    end

    function gpu.copy_from_device(device_ptr, buffer, byte_size)
        cuda.cuMemcpyDtoH(buffer, device_ptr[0], byte_size)
    end

    function gpu.run_kernel(kernel, tx,ty,tz, bx,by,bz, args)
        cuda.cuLaunchKernel(
            kernel, 
            tx,ty,tz, 
            bx,by,bz, 
            0, nil, args, nil
        )
    end

    function gpu.allocate_on_device(size, buffer)
        local ptr = ffi.new("void*[1]")
        cuda.cuMemAlloc(ptr, size)
        
        if buffer then
            cuda.cuMemcpyHtoD(ptr[0], buffer, size)
        end

        return ptr
    end
end

if CUBLAS then
    local ffi = require("ffi")

    -- Declare the necessary cuBLAS functions and types
    ffi.cdef[[
        typedef struct cublasContext *cublasHandle_t;
        
        int cublasCreate_v2(cublasHandle_t *handle);
        int cublasDestroy_v2(cublasHandle_t handle);
        int cublasSgemv_v2(
            cublasHandle_t handle, 
            int trans, 
            int m, 
            int n, 
            const float *alpha,
            const float *A, 
            int lda, 
            const float *x, 
            int incx, 
            const float *beta, 
            float *y, 
            int incy
        );
    ]]
    
    -- Load the cuBLAS library
    cublas = ffi.load("libcublas.so")
    
    -- Create a cuBLAS handle
    local handle = ffi.new("cublasHandle_t[1]")
    if cublas.cublasCreate_v2(handle) ~= 0 then
        error("Failed to create cuBLAS handle")
    end
    
    local function matmul(x, w, xout, n, d)
        -- Define alpha and beta for the operation
        local alpha = ffi.new("float[1]", 1.0)
        local beta = ffi.new("float[1]", 0.0)
    
        -- Call cuBLAS function sgemv to perform the matrix-vector multiplication
        local status = cublas.cublasSgemv(handle[0], cublas.CUBLAS_OP_T, n, d, alpha, w, n, x, 1, beta, xout, 1)
        if status ~= 0 then
            error("Failed to perform matrix-vector multiplication")
        end
    end

    gpu.matmulvec = matmul
end


do
    ffi.cdef[[
        typedef void *nvrtcProgram;
        const char* nvrtcGetErrorString(int result);
        int nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, 
                                    int numHeaders, const char** headers, const char** includeNames);
        int nvrtcDestroyProgram(nvrtcProgram* prog);
        int nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options);
        int nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet);
        int nvrtcGetPTX(nvrtcProgram prog, char* ptx);
        int nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet);
        int nvrtcGetProgramLog(nvrtcProgram prog, char* log);
    ]]

    local libnvrtc = ffi.load("nvrtc")
    nvrtc = setmetatable({}, {
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

    function gpu.compile_kernel(code, name)
        local module = ffi.new("CUmodule[1]")
        cuda.cuModuleLoadData(module, source_to_ptx(code))

        local kernel = ffi.new("CUfunction[1]")
        cuda.cuModuleGetFunction(kernel, module[0], name)
        assert(kernel[0] ~= nil)

        return kernel[0]
    end
end


return gpu