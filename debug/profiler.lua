local has_jit, jit_profiler = pcall(require, "debug.jit_profiler")
local profiler = {}
local should_run = true

function profiler.Start()
	if not should_run then return end

	jit_profiler.EnableStatisticalProfiling(true)
	jit_profiler.EnableTraceAbortLogging(true)
end

function profiler.Stop()
	if not should_run then return end

	local stats_filter = {
		{title = "all", filter = nil},
	}
	jit_profiler.EnableTraceAbortLogging(false)
	jit_profiler.EnableStatisticalProfiling(false)
	jit_profiler.PrintTraceAborts(10)

	for i, v in ipairs(stats_filter) do
		jit_profiler.PrintStatistical(10, v.title, v.filter)
	end
end

return profiler