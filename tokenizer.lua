local ffi = require("ffi")

local function Tokenizer(gguf_tokens, gguf_merges)
	local function Vocabulary()
		local tokens_map = {}

		for i, v in ipairs(gguf_tokens) do
			tokens_map[v] = i
		end

		return {
			get = function(i)
				return gguf_tokens[i]
			end,
			getIndex = function(str)
				return tokens_map[str]
			end,
			size = function()
				return #gguf_tokens
			end,
			token_map = tokens_map,
			tokens = gguf_tokens,
		}
	end

	local vocabulary = Vocabulary()
	local merges = {} -- passed to tokenizer
	do -- merge lines
		for k, v in ipairs(gguf_merges) do
			local l, r = v:match("(%S+) (%S+)")
			table.insert(merges, {vocabulary.getIndex(l), vocabulary.getIndex(r)})
		end
	end

	local all_tokens = vocabulary.size()
	local base_tokens = 128000
	local reservedSpecialTokens = all_tokens - base_tokens
	local specialTokensList = {}

	for i = base_tokens, all_tokens do
		table.insert(specialTokensList, vocabulary.get(i))
	end

	local specialTokens = {}

	for i = 1, #specialTokensList do
		specialTokens[specialTokensList[i]] = base_tokens + i - 1
	end

	-- actual tokenizer here
	local merges2 = {}

	for _, pair in ipairs(merges) do
		merges2[pair[1] .. "|" .. pair[2]] = vocabulary.getIndex(vocabulary.get(pair[1]) .. vocabulary.get(pair[2]))
	end

	local function encode2(str, special)
		if not special then encode_ordinary(str) end

		for k, v in pairs(special) do
			assert(specialTokens[k])
		end

		local specialPattern = {}

		for k, v in pairs(special) do
			table.insert(specialPattern, "\"" .. specialPattern .. "\"")
		end

		specialPattern = "(" .. table.concat(specialPattern, "|") .. ")"
	end

	local function encode(str)
		local function bytemap(b)
			if b <= 32 then
				return string.char(196, 128 + b)
			elseif b > 32 and b < 127 then
				return string.char(b)
			elseif b >= 127 and b <= 157 then
				return string.char(196, b - 127 + 161)
			elseif b > 157 and b <= 160 then
				return string.char(197, b - 158 + 128)
			elseif b == 173 then
				return string.char(197, 131)
			elseif b > 160 and b <= 191 then
				return string.char(194, b)
			elseif b > 191 then
				return string.char(195, b - 192 + 128)
			end

			error("byte out of range")
		end

		local function MERGE(str)
			local ids = {}

			table.sort(specialTokensList, function(a, b)
				return #a > #b
			end)

			local i = 1

			while i <= #str do
				local found = false

				if str:sub(i, i + 1) == "<|" then
					for k, v in pairs(specialTokensList) do
						if str:sub(i, i + #v - 1) == v then
							i = i + #v
							table.insert(ids, specialTokens[v])
							found = true

							break
						end
					end
				end

				if not found then
					table.insert(ids, vocabulary.getIndex(bytemap(str:byte(i))))
					i = i + 1
				end
			end

			while #ids > 2 do
				local stats = {}

				for i = 1, #ids - 1 do
					local key = ids[i + 0] .. "|" .. ids[i + 1]
					stats[key] = (stats[key] or 0) + 1
				end

				local minPair
				local minIndex = math.huge

				for pair, index in pairs(stats) do
					local mergeIndex = merges2[pair] or math.huge

					if mergeIndex < minIndex then
						minIndex = mergeIndex
						minPair = pair
					end
				end

				if not merges2[minPair] then break end

				local newIds = {}
				local i = 1

				while i <= #ids do
					if minPair == (ids[i] .. "|" .. (ids[i + 1] or "-")) then
						table.insert(newIds, merges2[minPair])
						i = i + 2
					else
						table.insert(newIds, ids[i])
						i = i + 1
					end
				end

				ids = newIds
			end

			return ids
		end

		local tokens = {}

		for i, v in ipairs(MERGE(str)) do
			tokens[i] = v
		end

		return tokens
	end

	return {
		encode = encode,
		vocabulary = vocabulary,
		merges = merges,
		specialTokens = specialTokens,
	}
end


if false then
    local tks = Tokenizer().encode("dæt ær øn står ære å snåkke mæd deg")
    local expected = {
        67,
        9371,
        83,
        66113,
        81,
        39218,
        77,
        357,
        18382,
        66113,
        265,
        13376,
        4224,
        3870,
        91861,
        296,
        9371,
        67,
        5367,
    }
    assert(#tks == #expected)

    for k, v in pairs(expected) do
        assert(tks[k] == expected[k] + 1)
    end
end

return Tokenizer