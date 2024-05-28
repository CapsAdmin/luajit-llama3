local ToppSampler = {}
ToppSampler.__index = ToppSampler

function ToppSampler:new(maxNumberOfElements, temperature, topp)
    local self = setmetatable({}, ToppSampler)
    self.indices = {}
    for i = 1, maxNumberOfElements do
        self.indices[i] = 0
    end
    self.topp = topp
    self.temperature = temperature
    return self
end

local function swap(array, from, to)
    local tmp = array[from]
    array[from] = array[to]
    array[to] = tmp
end

local function siftDown(array, from, n, comparator)
    local prev, next = from, nil
    while true do
        next = 2 * prev + 1
        if next >= n then break end
        local r = 2 * prev + 2
        if r < n and comparator(array[r], array[next]) < 0 then
            next = r
        end
        if comparator(array[next], array[prev]) < 0 then
            swap(array, prev, next)
            prev = next
        else
            break
        end
    end
end

function ToppSampler:SampleToken(logits)
    logits:DivideInPlace(0, logits.size, self.temperature)
    logits:SoftMaxInPlace(0, logits.size)

    local comparator = function(a, b)
        return logits:GetFloat(b) - logits:GetFloat(a)
    end

    local n = logits.size
    local head = 1
    local tail = n
    local cutoff = (1.0 - self.topp) / (n - 1)

    for i = 1, #self.indices do
        if logits:GetFloat(i - 1) >= cutoff then
            self.indices[head] = i - 1
            head = head + 1
        else
            self.indices[tail] = i - 1
            tail = tail - 1
        end
    end

    local n0 = head - 1

    for i = math.floor(n0 / 2), 1, -1 do
        siftDown(self.indices, i, n0, comparator)
    end

    local cumulativeProb = 0.0
    local lastIndex = 1

    for i = n0, 1, -1 do
        swap(self.indices, 1, i)
        cumulativeProb = cumulativeProb + logits:GetFloat(self.indices[i])
        if cumulativeProb > self.topp then
            lastIndex = i
            break
        end
        siftDown(self.indices, 1, i - 1, comparator)
    end

    local r = math.random() * cumulativeProb
    local cdf = 0.0

    for i = n0, lastIndex, -1 do
        cdf = cdf + logits:GetFloat(self.indices[i])
        if r < cdf then
            return self.indices[i]
        end
    end

    return self.indices[lastIndex]
end

return ToppSampler