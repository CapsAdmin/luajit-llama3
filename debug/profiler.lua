local has_jit, jit_profiler = pcall(require, "debug.jit_profiler")
if not has_jit then print(jit_profiler) end
local has_jit, trace_tracker = pcall(require, "debug.jit_trace_track")
if not has_jit then print(trace_tracker) end
local profiler = {}
local should_run = true

function profiler.Start()
	if not has_jit then return end

	jit_profiler.Start(
		{
			mode = "line",
			sampling_rate = 1,
			depth = 2, -- a high depth will show where time is being spent at a higher level in top level functions which is kinda useless
		}
	)
	trace_tracker.Start()
end

function profiler.Stop()
	if not has_jit then return end

	local traces, aborted = trace_tracker.Stop()
	print("\nluajit traces that were aborted and stitched:")
	print(trace_tracker.ToStringProblematicTraces(traces, aborted))
	print("\nprofiler statistics:")
	print(
		"I = interpreter, G = garbage collection, J = busy tracing, N = native / tracing completed:"
	)
	print(jit_profiler.Stop({sample_threshold = 100}))
end

return profiler
