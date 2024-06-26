local ffi = require("ffi")

local function initialize_cudart()
    local lib = ffi.load("libcudart.so")
    ffi.cdef[[
        const char* cudaGetErrorString(int error);
        int cudaFree(void* devPtr);
        int cudaDriverGetVersion(int *major);
        int cudaRuntimeGetVersion(int *major);
        int cudaMalloc(void **devPtr, size_t size);
    ]]
    
    local error_code = lib.cudaFree(nil)
    if error_code ~= 0 then
        local error_string = ffi.string(lib.cudaGetErrorString(error_code))
        error("cudaFree failed with error: " .. error_string)
    end

    local driverVersion = ffi.new("int[1]")
    local runtimeVersion = ffi.new("int[1]")
    lib.cudaDriverGetVersion(driverVersion)
    lib.cudaRuntimeGetVersion(runtimeVersion)

    print("CUDA Driver Version:", driverVersion[0])
    print("CUDA Runtime Version:", runtimeVersion[0])

    local N = 4
    local dA = ffi.new("float*[1]")
    res = lib.cudaMalloc(ffi.cast("void **", dA), N * N * ffi.sizeof("float"))
    print(res)
end

local function initialize_nvrtc(lib)
    local lib = ffi.load("libnvrtc.so")
    
    ffi.cdef[[
        typedef int nvrtcResult;
        nvrtcResult nvrtcVersion(int *major, int *minor);
        
        typedef void *nvrtcProgram;
        nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet);
        nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log);
        const char* nvrtcGetErrorString(nvrtcResult result);
        nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name,
                               int numHeaders, const char** headers, const char** includeNames);
        nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options);
        nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog);
        nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet);
        nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx);
    ]]
    local major = ffi.new("int[1]")
    local minor = ffi.new("int[1]")
    local result = lib.nvrtcVersion(major, minor)
    if result ~= 0 then
        error("nvrtcVersion failed with error code: " .. result)
    else
        print("NVRTC version: " .. major[0] .. "." .. minor[0])
    end

    local function compileKernel(source)
        local prog = ffi.new("nvrtcProgram[1]")
        local res = lib.nvrtcCreateProgram(prog, source, nil, 0, nil, nil)
        assert(res == 0, "Failed to create NVRTC program: " .. ffi.string(lib.nvrtcGetErrorString(res)))
    
        res = lib.nvrtcCompileProgram(prog[0], 0, nil)

        if res ~= 0 then
            local logSize = ffi.new("size_t[1]")
            lib.nvrtcGetProgramLogSize(prog[0], logSize)
            local log = ffi.new("char[?]", logSize[0])
            lib.nvrtcGetProgramLog(prog[0], log)
            error("Failed to compile NVRTC program: " .. ffi.string(log))
        end
    
        local ptxSize = ffi.new("size_t[1]")
        lib.nvrtcGetPTXSize(prog[0], ptxSize)
        local ptx = ffi.new("char[?]", ptxSize[0])
        lib.nvrtcGetPTX(prog[0], ptx)
        lib.nvrtcDestroyProgram(prog)
    
        return ffi.string(ptx, ptxSize[0])
    end

    assert(compileKernel([[
        __global__ void cuda_hello(){
            printf("Hello World from GPU!\n");
        }
    ]]):find("Generated by NVIDIA NVVM Compiler"))
end


local function check_nvrtc_compile()
    ffi.cdef[[
        typedef void *nvrtcProgram;
        nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet);
        nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log);
        const char* nvrtcGetErrorString(nvrtcResult result);
        nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name,
                               int numHeaders, const char** headers, const char** includeNames);
        nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char** options);
        nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog);
        nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet);
        nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx);
    ]]
    local lib = ffi.load("libnvrtc.so")
    local source = [[
        __global__ void cuda_hello(){
            printf("Hello World from GPU!\n");
        }
    ]]

    local prog = ffi.new("nvrtcProgram[1]")
    local res = lib.nvrtcCreateProgram(prog, source, nil, 0, nil, nil)
    assert(res == 0, "Failed to create NVRTC program: " .. ffi.string(lib.nvrtcGetErrorString(res)))

    res = lib.nvrtcCompileProgram(prog[0], 0, nil)

    if res ~= 0 then
        local logSize = ffi.new("size_t[1]")
        lib.nvrtcGetProgramLogSize(prog[0], logSize)
        local log = ffi.new("char[?]", logSize[0])
        lib.nvrtcGetProgramLog(prog[0], log)
        error("Failed to compile NVRTC program: " .. ffi.string(log))
    end

    local ptxSize = ffi.new("size_t[1]")
    lib.nvrtcGetPTXSize(prog[0], ptxSize)
    local ptx = ffi.new("char[?]", ptxSize[0])
    lib.nvrtcGetPTX(prog[0], ptx)
    lib.nvrtcDestroyProgram(prog)

    local ptx_code = ffi.string(ptx, ptxSize[0])

    assert(ptx_code:find("Generated by NVIDIA NVVM Compiler") ~= nil)
end

local function initialize_cuda(lib)
    local lib = ffi.load("libcuda.so")
    ffi.cdef[[
        int cuInit(unsigned int Flags);
        int cuGetErrorString(int error, const char **str);
        typedef struct CUctx_st *CUcontext;
        int cuCtxGetCurrent ( CUcontext* pctx );
    ]]

    local result = lib.cuInit(0)
    if result ~= 0 then
        error("cuInit failed with error code: " .. result)
    end

    local context = ffi.new("CUcontext[1]")
    local res = lib.cuCtxGetCurrent(context)

    if res == 0 and context[0] ~= nil then
        print("Current CUDA context is valid.")
    else
        local str = ffi.new("const char*[512]")
        print(lib.cuGetErrorString(res, ffi.cast("const char **", str)))
        print("No valid CUDA context found. Error:", ffi.string(str[0]))
    end
end

initialize_cudart()
initialize_nvrtc()
initialize_cuda()
check_nvrtc_compile()