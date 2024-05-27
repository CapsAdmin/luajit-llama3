
local function selectSampler(vocabularySize, temperature, topp, rngSeed)
	local sampler

	if temperature == 0 then
		sampler = function(logits)
			return logits:GetFloat(0)
		end
	else
		local innerSampler

		if topp <= 0 or topp >= 1 then
			innerSampler = function(logits)
				local random0to1 = math.random()
				local cdf = 0

				for i = 0, logits:Size() - 1 do
					cdf = cdf + logits:GetFloat(i)

					if random0to1 < cdf then return i end
				end

				return logits:Size() - 1
			end
		else
			local indices = {}
			local topp = topp
			local rngSeed = rngSeed
			innerSampler = function() end
		end

		sampler = function(logits)
			logits:DiviceInPlace(0, logits:size(), temperature)
			logits:SoftMaxInPlace(0, logits:Size())
			return innerSampler(logits)
		end
	end

	return sampler
end

local topp = 0.95


return selectSampler