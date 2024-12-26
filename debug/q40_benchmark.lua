require("debug.luajit_options").SetOptimized()
local measure = require("debug.measure_advanced")
local profiler = require("debug.profiler")
local Tensor = require("tensor"):UseComputeKernel("cpu_threads")
local gguf = require("gguf")
local metadata, tensors = gguf.load("/home/caps/projects/llama3.java/Meta-Llama-3-8B-Instruct-Q4_0.gguf")
local entry = tensors["blk.0.attn_output.weight"]
local wo = Tensor.New(entry.type_info.name, entry.size, entry.blob)

local function equal_around(a, b)
    local threshold = 0.00001
    if a == b then return true end
    if a > b-threshold and a < b+threshold then return true end
    error(a .. " ~= " .. b, 2)
end
if false then


measure("reading Q4_0 as single float", function() 
    local sum = 0
    for i = 0, wo.size - 1 do
        sum = sum + wo:GetFloat(i)
    end
    equal_around(sum, 14.916412353516)
end)
end
local size = 4096*8
local out = Tensor.New("F32", size)
local b = Tensor.New("F32", size)
b:FillInPlace(0, 4096, 1337)
local dim0 = size
local dim1 = size
local sum = 0

require("debug.profiler").Start()

measure("MatrixVectorMultiply", function()
    wo:MatrixVectorMultiply(b, out, dim0, dim1)
    --equal_around(out:GetFloat(1), -277.09658813477)
    --equal_around(out:GetFloat(2), -85.276184082031)
    --equal_around(out:GetFloat(132), -167.22700500488)
end)

require("debug.profiler").Stop()