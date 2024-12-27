local ffi = require("ffi")
local run_thread = function(ptr, udata)
	error("NYI")
end
local join_thread = function(id)
	error("NYI")
end

local get_cpu_threads = function(id)
	error("NYI")
end

if ffi.os == "Windows" then
	ffi.cdef[[
		typedef uint32_t (*thread_callback)(void*);

        void* CreateThread(
            void* lpThreadAttributes,
            size_t dwStackSize,
            thread_callback lpStartAddress,
            void* lpParameter,
            uint32_t dwCreationFlags,
            uint32_t* lpThreadId
        );
        uint32_t WaitForSingleObject(void* hHandle, uint32_t dwMilliseconds);
        int CloseHandle(void* hObject);
        uint32_t GetLastError(void);
        int32_t GetExitCodeThread(void* hThread, uint32_t* lpExitCode);

		typedef struct _SYSTEM_INFO {
            union {
                uint32_t dwOemId;
                struct {
                    uint16_t wProcessorArchitecture;
                    uint16_t wReserved;
                };
            };
            uint32_t dwPageSize;
            void* lpMinimumApplicationAddress;
            void* lpMaximumApplicationAddress;
            size_t dwActiveProcessorMask;
            uint32_t dwNumberOfProcessors;
            uint32_t dwProcessorType;
            uint32_t dwAllocationGranularity;
            uint16_t wProcessorLevel;
            uint16_t wProcessorRevision;
        } SYSTEM_INFO;

        void GetSystemInfo(SYSTEM_INFO* lpSystemInfo);
    ]]
	local kernel32 = ffi.load("kernel32")

	local function check_win_error(success)
		if success ~= 0 then return end

		local error_code = kernel32.GetLastError()
		local error_messages = {
			[5] = "Access denied",
			[6] = "Invalid handle",
			[8] = "Not enough memory",
			[87] = "Invalid parameter",
			[1455] = "Page file quota exceeded",
		}
		local err_msg = error_messages[error_code] or "unknown error"
		error(string.format("Thread operation failed: %s (Error code: %d)", err_msg, error_code), 2)
	end

	-- Constants
	local INFINITE = 0xFFFFFFFF
	local THREAD_ALL_ACCESS = 0x1F03FF

	function run_thread(func_ptr, udata)
		local thread_id = ffi.new("uint32_t[1]")
		local thread_handle = kernel32.CreateThread(
			nil, -- Security attributes (default)
			0, -- Stack size (default)
			ffi.cast("thread_callback", func_ptr),
			udata, -- Thread parameter
			0, -- Creation flags (run immediately)
			thread_id -- Thread identifier
		)

		if thread_handle == nil then check_win_error(0) end

		-- Return both handle and ID for Windows
		return {handle = thread_handle, id = thread_id[0]}
	end

	function join_thread(thread_data)
		local wait_result = kernel32.WaitForSingleObject(thread_data.handle, INFINITE)

		if wait_result == INFINITE then check_win_error(0) end

		local exit_code = ffi.new("uint32_t[1]")

		if kernel32.GetExitCodeThread(thread_data.handle, exit_code) == 0 then
			check_win_error(0)
		end

		if kernel32.CloseHandle(thread_data.handle) == 0 then check_win_error(0) end

		return exit_code[0]
	end

	function get_cpu_threads()
        local sysinfo = ffi.new("SYSTEM_INFO")
        kernel32.GetSystemInfo(sysinfo)
        return tonumber(sysinfo.dwNumberOfProcessors)
    end
else
	ffi.cdef[[
		typedef uint64_t pthread_t;

		typedef struct {
			uint32_t flags;
			void * stack_base;
			size_t stack_size;
			size_t guard_size;
			int32_t sched_policy;
			int32_t sched_priority;
		} pthread_attr_t;

		int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void *), void *arg);
		int pthread_join(pthread_t thread, void **value_ptr );

		long sysconf(int name);
	]]
	local pt = ffi.load("pthread")

	-- Enhanced pthread error checking
	local function check_pthread(int)
		if int == 0 then return end

		local error_messages = {
			[11] = "System lacks resources or reached thread limit",
			[22] = "Invalid thread attributes specified",
			[1] = "Insufficient permissions to set scheduling parameters",
			[3] = "Thread not found",
			[35] = "Deadlock condition detected",
			[12] = "Insufficient memory to create thread",
		}
		local err_msg = error_messages[int] or "unknown error"

		if err_msg then
			error(string.format("Thread operation failed: %s (Error code: %d)", err_msg, int), 2)
		end
	end

	function run_thread(func_ptr, udata)
		local thread_id = ffi.new("pthread_t[1]", 1)
		check_pthread(pt.pthread_create(thread_id, nil, func_ptr, udata))
		return thread_id[0]
	end

	function join_thread(id)
		check_pthread(pt.pthread_join(id, nil))
	end

	function get_cpu_threads()
		if ffi.os == "OSX" then return tonumber(ffi.C.sysconf(58)) end
	
		return tonumber(ffi.C.sysconf(83))
	end
end

local LUA_GLOBALSINDEX = -10002
ffi.cdef[[
    typedef struct lua_State lua_State;

    lua_State *luaL_newstate(void);
    void luaL_openlibs(lua_State *L);
    void lua_close(lua_State *L);
    int luaL_loadstring(lua_State *L, const char *s);
    int lua_pcall(lua_State *L, int nargs, int nresults, int errfunc);
    void lua_getfield(lua_State *L, int index, const char *k);
    void lua_settop(lua_State *L, int index);
    const char *lua_tolstring(lua_State *L, int index, size_t *len);
    ptrdiff_t lua_tointeger(lua_State *L, int index);
	int lua_gettop(lua_State *L);
    void lua_pushstring(lua_State *L, const char *s);
]]


local function create_lua_state()
	local L = ffi.C.luaL_newstate()

	if L == nil then error("Failed to create new Lua state: Out of memory", 2) end

	ffi.C.luaL_openlibs(L)
	return L
end

local function close_lua_state(L)
	ffi.C.lua_close(L)
end

local function check_error(L, ret)
	if ret == 0 then return end

	local chr = ffi.C.lua_tolstring(L, -1, nil)
	local msg = ffi.string(chr)
	ffi.C.lua_settop(L, -2)
	error(msg, 2)
end

local function run_code(L, code)
	check_error(L, ffi.C.luaL_loadstring(L, code))
	check_error(L, ffi.C.lua_pcall(L, 0, 1, 0))
end

local function get_function_pointer(L)
	ffi.C.lua_getfield(L, LUA_GLOBALSINDEX, "function_pointer")
	local func_ptr = ffi.C.lua_tointeger(L, -1)

	if func_ptr == 0 then error("cannot find function main", 2) end

	ffi.C.lua_settop(L, -2)
	return func_ptr
end

local thread_func_signature = "void *(*)(void *)"

local function load_thread(code)
	local L = create_lua_state()
	run_code(
		L,
		code .. [[
		_G.function_pointer = tonumber(ffi.cast('intptr_t', ffi.cast("void *(*)(void *)", callback)))
   		]]
	)
	return ffi.cast("void *(*)(void *)", get_function_pointer(L))
end

local function threaded_for(encode_code, lua_header_code, lua_code, thread_count)
	local struct = [[struct {
		int start;
		int stop;
		]] .. encode_code .. [[
	}]]
	local structcdata = ffi.typeof(struct)
	local struct_ptr = ffi.typeof(struct .. "*")
	local func_pool = {}
	local table_insert = table.insert
	local table_remove = table.remove
	local unpack = unpack
	local ipairs = ipairs

	-- we need at least thread_count amount of functions
	-- since multiple threads cannot access execute a lua function 
	-- in a single lua state at the same time
	for i = 1, thread_count do
		local func_ptr = load_thread(
			[==[
				package.path = './?.lua;' .. package.path
				local ffi = require("ffi")
				local udataptr = ffi.typeof([[]==] .. struct .. [==[*]])

			]==] .. lua_header_code .. [==[

				local function callback(udata)			
					local data = udataptr(udata)
					local start = data.start
					local stop = data.stop

					]==] .. lua_code .. [==[
				end
			]==]
		)
		func_pool[i] = func_ptr
	end

	return function(iterations, encode_callback, a, b, c, d, e, f, g, h, i, j, k, l, m, n)
		local chunks = iterations / thread_count
		local threads = {}
		local i = 0

		while i < iterations do
			local func_ptr = table_remove(func_pool)

			if not func_ptr then error("too few threads in thread pool", 2) end

			local tbl = {
				i,
				i + chunks,
			}
			encode_callback(tbl, a, b, c, d, e, f, g, h, i, j, k, l, m, n)
			table_insert(
				threads,
				{
					id = run_thread(func_ptr, structcdata(unpack(tbl))),
					func_ptr = func_ptr,
				}
			)
			i = i + chunks
		end

		for i, thread in ipairs(threads) do
			join_thread(thread.id)
			table_insert(func_pool, thread.func_ptr)
		end
	end
end

local function threaded_for2(callback, ctypes, thread_count, header_code)
	local bcode = string.dump(callback)
	local upvalues = {}

	for i = 1, math.huge do
		local name = debug.getlocal(callback, i)

		if not name then break end

		if i == 1 then
			assert(name == "thread_start")
		elseif i == 2 then
			assert(name == "thread_stop")
		else
			table.insert(upvalues, name)
		end
	end

	assert(#upvalues == #ctypes)
	local struct_code = ""
	local unpack_code = ""
	local encode_code = "local ffi = require('ffi')\n"

	for i = 1, #upvalues do
		local name = upvalues[i]
		local type = ctypes[i]

		if type == "@tensor" then
			struct_code = struct_code .. "void * " .. name .. ";\n"
			unpack_code = unpack_code .. "local " .. name .. " = Tensor:ThreadDeserialize(data." .. name .. ")\n"
			encode_code = encode_code .. "table.insert(data, " .. name .. ":ThreadSerialize())\n"
		elseif type == "@string" then
			struct_code = struct_code .. "uint8_t * " .. name .. ";\n"
			struct_code = struct_code .. "uint32_t " .. name .. "_len;\n"
			unpack_code = unpack_code .. "local " .. name .. " = ffi.string(data." .. name .. ", data." .. name .. "_len)\n"
			encode_code = encode_code .. "table.insert(data, ffi.cast('uint8_t*', " .. name .. "))\n"
			encode_code = encode_code .. "table.insert(data, #" .. name .. ")\n"
		elseif type == "@function" then
			struct_code = struct_code .. "uint8_t * " .. name .. ";\n"
			struct_code = struct_code .. "uint32_t " .. name .. "_len;\n"
			unpack_code = unpack_code .. "local " .. name .. " = loadstring(ffi.string(data." .. name .. ", data." .. name .. "_len, 'unpacking " .. name .. "'))\n"
			encode_code = encode_code .. "local bcode_" .. name .. " = string.dump(" .. name .. ")\n"
			encode_code = encode_code .. "table.insert(data, ffi.cast('uint8_t*', bcode_" .. name .. "))\n"
			encode_code = encode_code .. "table.insert(data, #bcode_" .. name .. ")\n"
		else
			struct_code = struct_code .. type .. " " .. name .. ";\n"
			unpack_code = unpack_code .. "local " .. name .. " = data." .. name .. "\n"
			encode_code = encode_code .. "table.insert(data, " .. name .. ")\n"
		end
	end

	local parallel_for = threaded_for(
		struct_code .. [[
			uint8_t *lua_bcode;
			uint32_t lua_bcode_len;
		]],
		[[
			_G.IS_THREAD = true
			require("debug.luajit_options").SetOptimized()
			local ffi = require("ffi")
			local Tensor = require("tensor"):UseComputeKernel("lua")

			local lua_func
		]],
		unpack_code .. [[
			lua_func = lua_func or loadstring(ffi.string(data.lua_bcode, data.lua_bcode_len), 'loadstring lua code')
			lua_func(start, stop, ]] .. table.concat(upvalues, ", ") .. [[)
		]],
		thread_count
	)
	local build_cdata = loadstring(
		[[
		local bcode = ...
		return function(data, ]] .. table.concat(upvalues, ", ") .. [[)
			]] .. encode_code .. [[
			
			table.insert(data, ffi.cast("uint8_t *", bcode))
			table.insert(data, #bcode)
		end
	]],
		"cdata build"
	)(bcode)
	return function(max, ...)
		parallel_for(max, build_cdata, ...)
	end
end

return {
	get_cpu_threads = get_cpu_threads,
	threaded_for = threaded_for2,
}
