local ffi = require("ffi")
local gpu = {}

local function bytes_to_gb(b)
	return string.format("%.2f", b / 1024 / 1024 / 1024)
end

local cuda
local nvrtc

do
	local header = [[
        typedef struct CUctx_st *CUcontext;
        typedef struct CUmod_st *CUmodule;
        typedef struct CUfunc_st *CUfunction;
        typedef struct CUstream_st *CUstream;
        typedef unsigned int CUdevice;

        int cuInit(unsigned int Flags);
        int cuDriverGetVersion ( int* driverVersion );
        int cuGetErrorString(int error, const char **str);
        int cuGetErrorName ( int error, const char ** pStr );

        int cuDeviceGetCount(int *count);
        int cuDeviceGet(CUdevice *device, int ordinal);
        int cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
        int cuDevicePrimaryCtxReset ( CUdevice dev );
        int cuDevicePrimaryCtxRelease ( CUdevice dev );
        int cuDevicePrimaryCtxGetState( CUdevice dev, unsigned int *flags, int *isactive);
        int cuDeviceGetAttribute( int* pi, unsigned int attrib, CUdevice dev );
        int cuDeviceGetName ( char* name, int len, CUdevice dev );
        int cuDeviceTotalMem ( size_t* bytes, CUdevice dev );

        int cuCtxSetCurrent ( CUcontext ctx );
        int cuCtxGetCurrent( CUcontext *ctx);
        int cuCtxDestroy ( CUcontext ctx );
        int cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
        int cuCtxSetLimit ( int limit, size_t value );
        int cuCtxGetLimit ( size_t* value, int limit );
        int cuCtxGetId ( CUcontext ctx, unsigned long long* ctxId );
        int cuCtxGetFlags ( unsigned int* flags );
        
        
        int cuModuleLoadData(CUmodule *module, const void *image);
        int cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
        int cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
        int cuMemAlloc_v2(void **dptr, size_t bytesize);
        int cuMemcpyHtoD_v2(void *dstDevice, const void *srcHost, size_t ByteCount);
        int cuMemcpyDtoH_v2(void *dstHost, const void *srcDevice, size_t ByteCount);
        int cuMemFree(void *dptr);
        int cuMemGetInfo_v2(size_t*free, size_t*total);
    ]]
	ffi.cdef(header)
	local cudaMemcpyHostToDevice = 1
	local cudaMemcpyDeviceToHost = 2
	local libcuda = ffi.load("libcuda")

	if false then
		local readelf = io.popen(
			"readelf -Ws --dyn-syms /nix/store/n91cvxk7dgf8ga03ac6mz48510awhpzp-nvidia-x11-550.54.14-6.7.7/lib/libcuda.so.550.54.14"
		):read("*all")

		for func_name in header:gmatch("(cu.-)%(") do
			func_name = func_name:gsub("%s+", "")

			for i = 1, 5 do
				if readelf:find(func_name .. "_v" .. i) then
					print(func_name .. " has a newer version " .. func_name .. "_v" .. i)
				end
			end
		end
	end

	cuda = setmetatable(
		{},
		{
			__index = function(s, key)
				local f = libcuda[key]
				return function(...)
					local res = f(...)

					if res ~= 0 then
						local desc = ffi.new("const char*[512]")
						libcuda.cuGetErrorString(res, ffi.cast("const char **", desc))
						desc = ffi.string(desc[0])
						local name = ffi.new("const char*[512]")
						libcuda.cuGetErrorName(res, ffi.cast("const char **", name))
						name = ffi.string(name[0])
						error(key .. " failed with " .. name .. "(" .. res .. "): " .. desc, 2)
					end
				end
			end,
		}
	)

	do -- init
		local function get_version()
			local out = ffi.new("int[1]")
			cuda.cuDriverGetVersion(out)
			out = out[0] / 1000
			return out
		end

		local function get_device_count()
			local out = ffi.new("int[1]")
			cuda.cuDeviceGetCount(out)
			return out[0]
		end

		local function get_device_id(dev)
			local out = ffi.new("int[0]")
			cuda.cuDeviceGet(out, dev)
			return out[0]
		end

		local function get_device_name(dev)
			local out = ffi.new("char[256]")
			cuda.cuDeviceGetName(out, 256, dev)
			return ffi.string(out)
		end

		local function get_device_total_mem(dev)
			local size = ffi.new("size_t[1]")
			cuda.cuDeviceTotalMem(size, dev)
			return tonumber(size[0])
		end

		local function get_device_attribute(dev, ...)
			local tr = {
				COMPUTE_CAPABILITY_MAJOR = 75,
				COMPUTE_CAPABILITY_MINOR = 76,
				MAX_THREADS_PER_BLOCK = 1,
				MAX_BLOCK_DIM_X = 2,
				MAX_BLOCK_DIM_Y = 3,
				MAX_BLOCK_DIM_Z = 4,
				MAX_GRID_DIM_X = 5,
				MAX_GRID_DIM_Y = 6,
				MAX_GRID_DIM_Z = 7,
			}
			local tbl = {}

			for i = 1, select("#", ...) do
				local out = ffi.new("int[1]")
				cuda.cuDeviceGetAttribute(out, tr[select(i, ...)], dev)
				tbl[i] = out[0]
			end

			return unpack(tbl)
		end

		local function get_current_context()
			local out = ffi.new("CUcontext[1]")
			cuda.cuCtxGetCurrent(out)
			return out[0]
		end

		local function set_current_context(context)
			cuda.cuCtxSetCurrent(context)
		end

		local function get_context_id(context)
			local out = ffi.new("unsigned long long[1]")
			cuda.cuCtxGetId(context, out)
			return tonumber(out[0])
		end

		local function get_context_limit(enum)
			local tr = {
				STACK_SIZE = 0x00,
				PRINTF_FIFO_SIZE = 0x01,
				MALLOC_HEAP_SIZE = 0x02,
				DEV_RUNTIME_SYNC_DEPTH = 0x03,
				DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
				MAX_L2_FETCH_GRANULARITY = 0x05,
				PERSISTING_L2_CACHE_SIZE = 0x06,
				SHMEM_SIZE = 0x07,
				CIG_ENABLED = 0x08,
				CIG_SHMEM_FALLBACK_ENABLED = 0x09,
			}
			local out = ffi.new("size_t[1]")
			cuda.cuCtxGetLimit(out, tr[enum])
			return tonumber(out[0])
		end

		local function create_device_context(dev)
			local context = ffi.new("CUcontext[1]")
			--cuda.cuDevicePrimaryCtxRetain(context, dev)
			local LMEM_RESIZE_TO_MAX = 0
			cuda.cuCtxCreate_v2(context, LMEM_RESIZE_TO_MAX, dev) -- this has a limit of 4gb for some reason
			return context[0]
		end

		local function get_device_primary_state(dev)
			local flags = ffi.new("int[1]")
			local state = ffi.new("int[1]")
			cuda.cuDevicePrimaryCtxGetState(dev, flags, state)
			return state[0] == 1
		end

		local function get_device_primary_flags(dev)
			local flags = ffi.new("int[1]")
			local state = ffi.new("int[1]")
			cuda.cuDevicePrimaryCtxGetState(dev, flags, state)
			return flags[0]
		end

		local function get_context_flags(dev)
			local flags = ffi.new("int[1]")
			cuda.cuCtxGetFlags(flags)
			return flags[0]
		end

		local function check_device_id(dev)
			local device_count = get_device_count()
			assert(
				device_count > 0 and dev < device_count,
				"device must be within range 0-" .. device_count
			)
			assert(get_device_id(dev) == dev, "invalid device id")
		end

		local function translate_context_flags(f)
			local out = {}
			local flags = {
				SCHED_AUTO = 0x00,
				SCHED_SPIN = 0x01,
				SCHED_YIELD = 0x02,
				SCHED_BLOCKING_SYNC = 0x04,
				BLOCKING_SYNC = 0x04,
				SCHED_MASK = 0x07,
				MAP_HOST = 0x08,
				LMEM_RESIZE_TO_MAX = 0x10,
				COREDUMP_ENABLE = 0x20,
				USER_COREDUMP_ENABLE = 0x40,
				SYNC_MEMOPS = 0x80,
			}

			for k, v in pairs(flags) do
				if bit.band(f, v) ~= 0 then table.insert(out, k) end
			end

			return out
		end

		function gpu.init_with_device(dev)
			print("cuda driver version: " .. get_version())
			cuda.cuInit(0)
			local name

			do -- device
				check_device_id(dev)
				print("using device: " .. get_device_name(dev))
				--print("\tvram: " .. bytes_to_gb(get_device_total_mem(dev)) .. "gb vram")
				local major, minor = get_device_attribute(dev, "COMPUTE_CAPABILITY_MAJOR", "COMPUTE_CAPABILITY_MINOR")
				print("\tcompute capability: " .. major .. "." .. minor)
				print("\tmax threads per block: " .. get_device_attribute(dev, "MAX_THREADS_PER_BLOCK"))
				local x, y, z = get_device_attribute(dev, "MAX_BLOCK_DIM_X", "MAX_BLOCK_DIM_Y", "MAX_BLOCK_DIM_Z")
				print("\tmax block dim: " .. x .. "-" .. y .. "-" .. z)
				local x, y, z = get_device_attribute(dev, "MAX_GRID_DIM_X", "MAX_GRID_DIM_Y", "MAX_GRID_DIM_Z")
				print("\tmax grid dim: " .. x .. "-" .. y .. "-" .. z)
			end

			do -- context
				assert(get_current_context() == nil, "context already set")
				local context = create_device_context(dev)
				-- additional checks
				set_current_context(context)
				assert(get_current_context() == context, "context is not properly set")
			--print("using context: " .. get_context_id(context))
			--print("\tmalloc heap size: " .. bytes_to_gb(get_context_limit("MALLOC_HEAP_SIZE")))
			--print("\twith ctx flags: " .. table.concat(translate_context_flags(get_context_flags(dev)), " "))
			--print("\twith primary ctx flags: " .. table.concat(translate_context_flags(get_device_primary_flags(dev)), " "))
			end

			local left, total = gpu.get_memory()
			print("\t" .. bytes_to_gb(left) .. "gb left out of " .. bytes_to_gb(total) .. "gb")
		end
	end

	function gpu.get_memory()
		local free = ffi.new("size_t[1]")
		local available = ffi.new("size_t[1]")
		cuda.cuMemGetInfo_v2(free, available)
		return tonumber(free[0]), tonumber(available[0])
	end

	function gpu.copy_to_device(device_ptr, buffer, byte_size)
		cuda.cuMemcpyHtoD_v2(device_ptr[0], buffer, byte_size)
	end

	function gpu.copy_from_device(device_ptr, buffer, byte_size)
		cuda.cuMemcpyDtoH_v2(buffer, device_ptr[0], byte_size)
	end

	function gpu.run_kernel(kernel, tx, ty, tz, bx, by, bz, args)
		cuda.cuLaunchKernel(kernel, tx, ty, tz, bx, by, bz, 0, nil, args, nil)
	end

	function gpu.allocate_on_device(size, buffer)
		local ptr = ffi.new("void*[1]")
		cuda.cuMemAlloc_v2(ptr, size)

		if buffer then gpu.copy_to_device(ptr, buffer, size) end

		local left, total = gpu.get_memory()
		print(
			"gpu allocation " .. bytes_to_gb(size) .. "gb - " .. bytes_to_gb(left) .. "gb left out of " .. bytes_to_gb(total) .. "gb"
		)
		return ptr
	end
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
	nvrtc = setmetatable(
		{},
		{
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
						error(
							key .. " failed with code (" .. res .. "): " .. ffi.string(libnvrtc.nvrtcGetErrorString(res)) .. "\n" .. log,
							2
						)
					end
				end
			end,
		}
	)

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