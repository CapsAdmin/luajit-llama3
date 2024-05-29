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

local function load_thread(header, code)
	local L = create_lua_state()
	run_code(
		L,
		header .. [[local ffi = require("ffi");local function callback(udata) ]] .. code .. [[
      end
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

local function inject_threaded_matmul(Tensor)
	local struct = [[struct {
		int start; 
		int stop; 
		int dim1; 
		
		void *out; 
		void *self;
		void *that;
	}]]
	local structcdata = ffi.typeof(struct)
	local struct_ptr = ffi.typeof(struct .. "*")
	local thread_pool = {}

	for i = 1, 32 do
		local func_ptr = load_thread(
			[==[
				local Tensor = require("tensor")
				local ffi = require("ffi")
				local udataptr = ffi.typeof([[]==] .. struct .. [==[*]])
			]==],
			[==[
				local data = udataptr(udata)
				
				local dim1 = data.dim1
				local out = Tensor:ThreadDeserialize(data.out)
				local self = Tensor:ThreadDeserialize(data.self)
				local that = Tensor:ThreadDeserialize(data.that)

				for i = data.start, data.stop-1 do
					out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
				end
			]==]
		)
		thread_pool[i] = func_ptr
	end

	function Tensor:MatMul(that, out, dim0, dim1)
		local chunks = dim0 / 32
		local threads = {}
		local i = 0

		while i < dim0 do
			local start = i
			local stop = i + chunks
			local func_ptr = assert(table.remove(thread_pool))
			local thread_id = run_thread(
				func_ptr,
				structcdata(
					{
						start = start,
						stop = stop,
						dim1 = dim1,

						out = out:ThreadSerialize(),
						self = self:ThreadSerialize(),
						that = that:ThreadSerialize(),
					}
				)
			)
			table.insert(threads, thread_id)
			i = i + chunks
			table.insert(thread_pool, 1, func_ptr)
		end

		for i, thread_id in ipairs(threads) do
			check_pthread(pt.pthread_join(thread_id, nil))
		end
	end
end

return inject_threaded_matmul