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

local function run_threaded(code, udata)
	local L = create_lua_state()
	run_code(
		L,
		[[local ffi = require("ffi");local function callback(udata) ]] .. code .. [[
      end
      _G.function_pointer = tonumber(ffi.cast('intptr_t', ffi.cast("void *(*)(void *)", callback)))
   ]]
	)
	local func_ptr = ffi.cast("void *(*)(void *)", get_function_pointer(L))
	local thread_id = ffi.new("pthread_t[1]", 1)
	check_pthread(pt.pthread_create(thread_id, nil, func_ptr, udata))
	return thread_id[0]
end

local function multithreading(n_threads, code, udata)
	local threads = {}

	-- run threads
	for i = 0, n_threads - 1 do
		threads[i + 1] = run_threaded(code, udata)
	end

	-- wait for all to finish
	for _, thread_id in ipairs(threads) do
		check_pthread(pt.pthread_join(thread_id, nil))
	end
end

do -- test for MatMul
	local Tensor = require("tensor")
	local size = 16
	local t1 = Tensor:F32(size * size)
	local t2 = Tensor:F32(size * size)
	local out = Tensor:F32(size)

	for i = 0, (size * size) - 1 do
		t1:SetFloat(i, i)
	end

	for i = 0, size * size - 1 do
		t2:SetFloat(i, i)
	end

	t1:MatMul(t2, out, size, size)
	local expected = {}

	for i = 0, size - 1 do
		expected[i] = out:GetFloat(i)
	end

	function Tensor:MatMul2(that, out, dim0, dim1)
		local chunks = dim0 / 4
		local threads = {}

		-- run threads
		for i = 0, chunks - 1 do
			threads[i + 1] = run_threaded(
				[[
				local Tensor = require("tensor")
				local out = Tensor:F32(16)
				local self = Tensor:F32(16*16)
				local that = Tensor:F32(16*16)
				local data = ffi.cast("struct {int start; int stop; int dim1; float *out_blob; float *self_blob; float * that_blob;}*", udata)
				local dim1 = data.dim1

				out.blob = data.out_blob
				self.blob = data.self_blob
				that.blob = data.that_blob

				for i = data.start, data.stop - 1 do
					out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
				end
			]],
				ffi.new(
					"struct {int start; int stop; int dim1; float *out_blob; float *self_blob; float *that_blob;}",
					{
						start = i * chunks,
						stop = i * chunks + 4,
						dim1 = dim1,
						out_blob = out.blob,
						self_blob = self.blob,
						that_blob = that.blob,
					}
				)
			)
		end

		for _, thread_id in ipairs(threads) do
			check_pthread(pt.pthread_join(thread_id, nil))
		end
	end

	local out = Tensor:F32(size)
	t1:MatMul2(t2, out, size, size)

	for i = 0, size - 1 do
		if expected[i] ~= out:GetFloat(i) then
			print(expected[i], out:GetFloat(i))
		end
	end
end

--[[
   function Tensor:MatMul(that, out, dim0, dim1)
      for i = 0, dim0 - 1 do
         out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
      end
   end
]] do
	return
end

local buffer = ffi.new("float[16]")
multithreading(
	16,
	[[
         local Tensor = require("tensor")
         local t2 = Tensor:F32(10)
         
         -- test
         t2.blob = ffi.cast("float *", udata)
         t2.size = 10

         local buffer = ffi.cast("float *", udata)
         buffer[4] = 1
      ]],
	buffer
)

for i = 0, 16 - 1 do
	print(buffer[i])
end