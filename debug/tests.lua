do -- some tensor tests
    local Tensor = require("tensor")
    local t = Tensor.New("F32", 10)

    for i = 0, 10 - 1 do
        t:SetFloat(i, i)
    end

    for i = 0, 10 - 1 do
        assert(t:GetFloat(i) == i)
    end

    assert(t:Sum(0, t.size) == 45)
    assert(t:Max(0, t.size) == 9)
    local t2 = Tensor.New("F32", 10)
    t:CopyTo(0, t2, 0, t.size)

    for i = 0, 10 - 1 do
        assert(t2:GetFloat(i) == i)
    end

    for i = 0, 10 - 1 do
        t2:SetFloat(i, 0)
    end

    t:CopyTo(5, t2, 0, 5)

    for i = 5, 9 do
        assert(t2:GetFloat(i - 5) == i)
    end

    do
        local size = 10
        local t1 = Tensor.New("F32", size)
        local t2 = Tensor.New("F32", size)

        for i = 0, size - 1 do
            t1:SetFloat(i, i)
            t2:SetFloat(i, i * 2)
        end

        local dot_product = t1:Dot(0, t2, 0, size)
        local expected_dot_product = 0

        for i = 0, size - 1 do
            expected_dot_product = expected_dot_product + i * (i * 2)
        end

        assert(expected_dot_product == dot_product)
    end

    do -- test for MatrixVectorMultiply
        local size = 3
        local t1 = Tensor.New("F32", size * size)
        local t2 = Tensor.New("F32", size * size)
        local out = Tensor.New("F32", size)
        t1:SetFloat(0, 1)
        t1:SetFloat(1, 2)
        t1:SetFloat(2, 3)
        t1:SetFloat(3, 4)
        t1:SetFloat(4, 5)
        t1:SetFloat(5, 6)
        t1:SetFloat(6, 7)
        t1:SetFloat(7, 8)
        t1:SetFloat(8, 9)
        t2:SetFloat(0, 1)
        t2:SetFloat(1, 0)
        t2:SetFloat(2, 0)
        t2:SetFloat(3, 0)
        t2:SetFloat(4, 1)
        t2:SetFloat(5, 0)
        t2:SetFloat(6, 0)
        t2:SetFloat(7, 0)
        t2:SetFloat(8, 1)
        t1:MatrixVectorMultiply(t2, out, size, size)
        assert(out:GetFloat(0) == 1)
        assert(out:GetFloat(1) == 4)
        assert(out:GetFloat(2) == 7)
    end
end

do -- blob
    local Tensor = require("tensor")
    local ffi = require("ffi")


    do -- q4_0
        local block_size = 32

        local function mock_q40_blob(size)
            local b = Tensor.New("Q4_0", size)
            for i = 0, b.size - 1 do
                b.blob[i] = i % 255
            end
            return b
        end

        local t = mock_q40_blob(512)

        local function check(sum)
            local expected = "10403395.617538"
            if tostring(sum) ~= expected then
                error("Q4_0 read failed, expected " .. expected .. " got " .. tostring(sum), 2)
            end
        end

        do
            local sum = 0
            for i = 0, t.size-1 do 
                sum = sum + t:GetFloat(i)
            end
            check(sum)
        end
    end

    do
        local ffi = require("ffi")
        local b = Tensor.New("F32", 10)
        b:FillInPlace(0, b.size, 1337)
        for i = 0, b.size - 1 do
            assert(b:GetFloat(i) == 1337)
        end
    end

    do
        local ffi = require("ffi")
        local b = Tensor.New("F32", 10)
        b:FillInPlace(0, b.size, 0)
        for i = 0, b.size - 1 do
            assert(b:GetFloat(i) == 0)
        end
    end
end

require("debug.cpu_matrix_benchmark")
require("debug.cpu_plain_matrix_benchmark")
require("debug.q40_benchmark")
--require("debug.gpu_matrix_benchmark")