local ffi = require("ffi")
local Tensor = require("tensor")
require("tensor_compute_ext").use_cuda()

function Tensor:MatrixVectorMultiplyCPU(that, out, dim0, dim1)
	for i = 0, dim0 - 1 do
		local result = 0

		for j = 0, dim1 - 1 do
			result = result + self:GetFloat(i * dim1 + j) * that:GetFloat(j)
		end

		out:SetFloat(i, result)
	end
end


local variants = {
	{
		out = Tensor:F32(4096),
		a = Tensor:F32(16777216),
		b = Tensor:F32(4096),
		dim0 = 4096,
		dim1 = 4096,
	},
	{
		out = Tensor:F32(4096),
		a = Tensor:F32(4194304),
		b = Tensor:F32(4096),
		dim0 = 1024,
		dim1 = 4096,
	},
	{
		out = Tensor:F32(14336),
		a = Tensor:F32(58720256),
		b = Tensor:F32(4096),
		dim0 = 14336,
		dim1 = 4096,
	},
	{
		out = Tensor:F32(4096),
		a = Tensor:F32(58720256),
		b = Tensor:F32(14336),
		dim0 = 4096,
		dim1 = 14336,
	},
	{
		out = Tensor:F32(128256),
		a = Tensor:F32(525336576),
		b = Tensor:F32(4096),
		dim0 = 128256,
		dim1 = 4096,
	},
}

math.randomseed(1337)
for k, v in ipairs(variants) do
    v.a:MapInPlace(0, v.a.size, function(v, i) return math.random()*2-1 end)
    v.b:MapInPlace(0, v.b.size, function(v, i) return math.random()*2-1 end)
    v.out.blob:Fill(0)
end

for k, v in ipairs(variants) do
    v.a:MatrixVectorMultiplyCPU(v.b, v.out, v.dim0, v.dim1)
    local expected = v.out
    local out = Tensor:F32(v.out.size)
    out.blob:Fill(0)
    
    v.a:MatrixVectorMultiply(v.b, out, v.dim0, v.dim1)
    local fail = false
    for i = 0, v.dim0 - 1 do
        local a = expected:GetFloat(i)
        local b = out:GetFloat(i)

        if (b - a) > 0.01 then
            print(v.a, "failed at index", i)
            print("expected " .. a .. " got " .. b)
            print("diff: " .. (a - b))
            fail = true
        end
    end
    if fail then break end
end