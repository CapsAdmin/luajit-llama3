local ffi = require("ffi")
local Tensor = {}
Tensor.__index = Tensor

function Tensor:new(size, blob, blob_ref, get_float, set_float)
    local t = setmetatable({}, Tensor)
    assert(type(size) == "number" or type(size) == "cdata") -- ULL
    assert(type(blob) == "cdata")
    assert(blob_ref)
    assert(type(set_float) == "function")
    assert(type(get_float) == "function")

    t.blob = blob
    t.size = size
    t.SetFloat = set_float
    t.GetFloat = get_float
    t.blob_ref = blob_ref -- need to keep this around

    return t
end

local ggf = require("gguf")
do
    local function get_float(self, index)
        assert(index >= 0)
        assert(index < self.size)
    
        return self.blob[index]
    end
    
    local function set_float(self, index, val)
        assert(index >= 0)
        assert(index < self.size)
    
        self.blob[index] = val
    end

    function Tensor:F32(size, blob)
        local blob_ref = blob

        if not blob then
            blob = ffi.new("float[?]", size)
            blob_ref = blob
        end

        return Tensor:new(size, ffi.cast("float *", blob), blob_ref, get_float, set_float)
    end
end

do
    local block_size = ggf.GGMLTypeMap.Q4_0.blockSize
    local type_size = ggf.GGMLTypeMap.Q4_0.typeSize
    local FLOAT16 = 2

    local function get_float(self, index)
        assert(index >= 0)
        assert(index < self.size)
    
        local block_index = index / block_size
        local block_offset = block_index * type_size
        
        local blob = self.blob
    
        local scale = blob[index] -- read uint16_t
        local quant
        local modIndex = index % block_size
    
        if modIndex < block_size / 2 then
            quant = bit.band(blob[block_offset + FLOAT16 + modIndex], 0x0F)
        else
            quant = bit.band(
                bit.rshift(blob[block_offset + FLOAT16 + modIndex - block_size / 2], 4),
                0x0F
            )
        end
    
        quant = quant - 8
        return quant * scale
    end

    local function set_float(self, index, value)
        assert(index >= 0)
        assert(index < self.size)
        
        error("NYI", 2)
    end 

    function Tensor:Q4_0(size, blob)
        local blob_ref = blob

        if not blob then
            blob = ffi.new("uint16_t[?]", size)
            blob_ref = blob
        end

        return Tensor:new(size, ffi.cast("uint16_t *", blob), blob_ref, get_float, set_float)
    end
end

function Tensor:GetFloatVector(index, value)
    error("NYI")
end

function Tensor:ScalarDot(thisOffset, that, thatOffset, size)
    local result = 0
    for j = 0, size - 1 do
        result = result + self:GetFloat(thisOffset + j) * that:GetFloat(thatOffset + j)
    end

    return result
end

function Tensor:Dot(thisOffset, that, thatOffset, size)
    return self.ScalarDot(self, thisOffset, that, thatOffset, size)
end

function Tensor:MatMul(that, out, dim0, dim1)
    for i = 0, dim0 - 1 do
        out:SetFloat(i, self:Dot(i * dim1, that, 0, dim1))
    end
end

function Tensor:Reduce(thisOffset, size, seed, reduce_callback)
    local result = seed

    for i = 0, size-1 do
        result = reduce_callback(result, self:GetFloat(thisOffset + i))
    end

    return result
end

function Tensor:Sum(thisOffset, size)
    return self:Reduce(thisOffset, size, 0, function(r, f)
        return r + f
    end)
end

function Tensor:Max(thisOffset, size)
    return self:Reduce(thisOffset, size, 0, function(r, f)
        return math.max(r, f)
    end)
end

function Tensor:CopyTo(thisOffset, that, thatOffset, size)
    return that:MapWithIndexInPlace(thatOffset, size, function(value, index) return self:GetFloat(index - thatOffset + thisOffset) end);
end

function Tensor:MapWithIndexInPlace(thisOffset, size, mapWithIndexFunction)
    local endOffset = thisOffset + size;
    for i = thisOffset, endOffset-1 do
        self:SetFloat(i, assert(mapWithIndexFunction(assert(self:GetFloat(i)), i)))
    end
    return self
end

do -- some tests
    local ggf = require("gguf")
    
    local t = Tensor:F32(10)
    for i = 0, 10-1 do
        t:SetFloat(i, i)
    end

    for i = 0, 10-1 do
        assert(t:GetFloat(i) == i)
    end

    assert(t:Sum(0, t.size) == 45)
    assert(t:Max(0, t.size) == 9)

    local t2 = Tensor:F32(10)
    t:CopyTo(0, t2, 0, t.size)
    for i = 0, 10-1 do
        assert(t2:GetFloat(i) == i)
    end

    for i = 0, 10-1 do
        t2:SetFloat(i, 0)
    end

    t:CopyTo(5, t2, 0, 5)
    
    for i = 5, 9 do
        assert(t2:GetFloat(i-5) == i)
    end

    do
        local size = 10
        local t1 = Tensor:F32(size)
        local t2 = Tensor:F32(size)
        
        for i = 0, size-1 do
            t1:SetFloat(i, i)
            t2:SetFloat(i, i*2)
        end
        
        local dot_product = t1:Dot(0, t2, 0, size)
        local expected_dot_product = 0
        for i = 0, size-1 do
            expected_dot_product = expected_dot_product + i * (i * 2)
        end

        assert(expected_dot_product == dot_product)
    end
end

return Tensor