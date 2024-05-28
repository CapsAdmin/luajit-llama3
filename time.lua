local ffi =require("ffi")
ffi.cdef([[
		struct timespec {
			long int tv_sec;
			long tv_nsec;
		};
		int clock_gettime(int clock_id, struct timespec *tp);
	]])
	local ts = ffi.new("struct timespec")
	local enum = 1
	local func = ffi.C.clock_gettime

	local function get_time()
    func(enum, ts)
    return tonumber(ts.tv_sec) + tonumber(ts.tv_nsec) * 0.000000001
end

return get_time