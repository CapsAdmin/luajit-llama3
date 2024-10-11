local measure = require("debug.measure")
local profiler = require("debug.profiler")
local Tensor = require("tensor")
Tensor:UseComputeKernel("lua")
local gguf = require("gguf")
local metadata, tensors = gguf.load("/home/caps/projects/llama3.java/Meta-Llama-3-8B-Instruct-Q4_0.gguf")
local entry = tensors["blk.0.attn_output.weight"]
local wo = Tensor[entry.type_info.name](Tensor, entry.size, entry.blob)


measure("reading Q4_0 as float block")
local sum = 0
for _ = 1, 10 do
    for i = 0, (wo.size/32) - 1 do
        local floats = wo.blob.Get32FloatsFromBlockIndex(i)
        for j = 0, 31 do
            sum = sum + floats[j]
        end
    end
end
assert(math.floor(sum) == 149)
measure()


measure("reading Q4_0 as single float")
local sum = 0
for _ = 1, 10 do
    for i = 0, wo.size - 1 do
        sum = sum + wo.blob.GetFloat(i)
    end
end
assert(math.floor(sum) == 149)
measure()