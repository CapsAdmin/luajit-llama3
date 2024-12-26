local ffi = require("ffi")
local pt = ffi.load("pthread")
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
]]

local function get_cpu_threads()
	if ffi.os == "OSX" then return tonumber(ffi.C.sysconf(58)) end

	return tonumber(ffi.C.sysconf(83))
end

local function create_lua_state()
	local L = assert(ffi.C.luaL_newstate())
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

local EAGAIN = 11
local EINVAL = 22
local EPERM = 1

local function check_pthread(int)
	if int == EAGAIN then
		error("try again", 2)
	elseif int == EINVAL then
		error("invalid settings", 2)
	elseif int == EPERM then
		error("no permission", 2)
	end
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

local function run_thread(func_ptr, udata)
	local thread_id = ffi.new("pthread_t[1]", 1)
	check_pthread(pt.pthread_create(thread_id, nil, func_ptr, udata))
	return thread_id[0]
end

local function join_thread(id)
	check_pthread(pt.pthread_join(id, nil))
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
			unpack_code = unpack_code .. "local " .. name .. " = loadstring(ffi.string(data." .. name .. ", data." .. name .. "_len, 'unpacking "..name.."'))\n"
			encode_code = encode_code .. "local bcode_"..name.." = string.dump("..name..")\n"
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
	"cdata build")(bcode)
	return function(max, ...)
		parallel_for(max, build_cdata, ...)
	end
end

return {
	get_cpu_threads = get_cpu_threads,
	threaded_for = threaded_for2,
}