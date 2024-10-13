local get_time = require("debug.get_time")

local function get_median(tbl, start, stop)
	start = start or 1
	stop = stop or #tbl
	local new = {}

	for i = start, stop do
		table.insert(new, tbl[i])
	end

	table.sort(new)
	local median = new[math.ceil(#new / 2)] or new[1]
	return median
end

local function get_average(tbl, start, stop)
	start = start or 1
	stop = stop or #tbl

	if #tbl == 0 then return nil end

	local n = 0
	local count = 0

	for i = start, stop do
		n = n + tbl[i]
		count = count + 1
	end

	return n / count
end

return function(what, cb) -- type util.Measure = function(string, function): any
	local space = (" "):rep(40 - #what)
	io.write("> ", what, "\n")
	local times = {}
	local threshold = 0.01
	local lookback = 5

	for i = 1, 30 do
		local time = get_time()
		local ok, err = pcall(cb)
		times[i] = get_time() - time
		io.write(("%.5f"):format(times[i]), " seconds\n")

		if i >= lookback and times[i] > 0.5 then
			local current = get_average(times)
			local latest = get_average(times, #times - lookback + 1)
			local diff = math.abs(current - latest)

			if diff > 0 and diff < threshold then
				io.write(
					"time difference the last ",
					lookback,
					" times (",
					diff,
					") met the threshold (",
					threshold,
					"), stopped measuring.\n"
				)

				break
			end
		end

		if not ok then
			io.write(" - FAIL: ", err)
			error(err, 2)
		end
	end

	local average = get_average(times)
	local median = get_median(times)
	table.sort(times)
	local min = times[1]
	local max = times[#times]
	io.write(
		"< FINISHED: ",
		("%.5f"):format(median),
		" seconds (median), ",
		("%.5f"):format(average),
		" seconds (average)\n"
	)
end