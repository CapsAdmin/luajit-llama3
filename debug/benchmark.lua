local measure = require("debug.measure")
local Tensor = require("tensor")

local size = 100000
local t1 = Tensor:F32(size*size)
local t2 = Tensor:F32(size*size)
local out = Tensor:F32(size)

--t1:MapInPlace(0, t1.size, function(v,i) return i end)
--t2:MapInPlace(0, t2.size, function(v,i) return i end)

Tensor:EnableThreadedMatrixDotProduct()
for i = 1, 10 do
    measure("MatrixDotProduct multi threaded")
    t1:MatrixDotProduct(t2, out, size, size)
    measure()
end