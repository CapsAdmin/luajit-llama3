local measure = require("debug.measure")
local profiler = require("debug.profiler")
local Weights = require("weights")
local Configuration = require("configuration")
local ggf = require("gguf")
local gguf = ggf.load_gguf("/home/caps/projects/llama3.java/Meta-Llama-3-8B-Instruct-Q4_0.gguf")
local weights = Weights(gguf.tensors, gguf.metadata["llama.block_count"])
local wo = weights.wo[1]


measure("reading Q4_0")
local sum = 0
for _ = 1, 10 do
    for i = 0, wo.size-1 do
        sum = sum + wo:GetFloat(i)
    end
end
assert(math.floor(sum) == -231)
measure()