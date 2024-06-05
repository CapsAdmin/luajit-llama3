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

local function sift_down(logits, array, from, n)
	local prev, next = from, nil

	while true do
		next = 2 * prev + 1

		if next >= n then break end

		local r = 2 * prev + 2

		if r < n and logits:GetFloat(array[r]) - logits:GetFloat(array[next]) < 0 then
			next = r
		end

		if logits:GetFloat(array[next]) - logits:GetFloat(array[prev]) < 0 then
			array[prev], array[next] = array[next], array[prev]
			prev = next
		else
			break
		end
	end
end

function ToppSampler:SampleToken(logits)
	logits:DivideInPlace(0, logits.size, self.temperature)
	logits:SoftMaxInPlace(0, logits.size)
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
		sift_down(logits, self.indices, i, n0)
	end

	local cumulativeProb = 0.0
	local lastIndex = 1

	for i = n0, 1, -1 do
		self.indices[1], self.indices[i] = self.indices[i], self.indices[1]
		cumulativeProb = cumulativeProb + logits:GetFloat(self.indices[i])

		if cumulativeProb > self.topp then
			lastIndex = i

			break
		end

		sift_down(logits, self.indices, 1, i - 1)
	end

	local r = math.random() * cumulativeProb
	local cdf = 0.0

	for i = n0, lastIndex, -1 do
		cdf = cdf + logits:GetFloat(self.indices[i])

		if r < cdf then return self.indices[i] end
	end

	return self.indices[lastIndex]
end

return ToppSampler