local TOKEN_MERGE_SEPARATOR = "|"
local BASE_TOKEN_INDEX = 128000
local Tokenizer = {}
Tokenizer.__index = Tokenizer

function Tokenizer:new(gguf_tokens, gguf_merges)
	local index_to_token = gguf_tokens
	local token_to_index = {}

	for i, v in ipairs(gguf_tokens) do
		token_to_index[v] = i
	end

	local merges = {}

	for _, v in ipairs(gguf_merges) do
		local l, r = v:match("(%S+) (%S+)")
		local l_index = token_to_index[l]
		local r_index = token_to_index[r]
		merges[l_index .. TOKEN_MERGE_SEPARATOR .. r_index] = token_to_index[index_to_token[l_index] .. index_to_token[r_index]]
	end

	local special_token_list = {}
	local special_tokens = {}

	do
		for i = BASE_TOKEN_INDEX, #gguf_tokens do
			table.insert(special_token_list, index_to_token[i])
		end

		for i = 1, #special_token_list do
			special_tokens[special_token_list[i]] = BASE_TOKEN_INDEX + i - 1
		end

		table.sort(special_token_list, function(a, b)
			return #a > #b
		end)
	end

	return setmetatable(
		{
			merges = merges,
			index_to_token = index_to_token,
			token_to_index = token_to_index,
			special_token_list = special_token_list,
			special_tokens = special_tokens,
		},
		Tokenizer
	)
end

do
	local function reverse_bytemap(b1, b2)
		if b1 == 196 then
			if b2 >= 128 and b2 <= 160 then
				return b2 - 128, 2
			elseif b2 >= 161 and b2 <= 191 then
				return b2 - 161 + 127, 2
			end
		elseif b1 == 197 then
			if b2 >= 128 and b2 <= 130 then
				return b2 + 158, 2
			elseif b2 == 131 then
				return 173, 2
			end
		elseif b1 == 194 then
			if b2 >= 161 and b2 <= 191 then return b2, 2 end
		elseif b1 == 195 then
			if b2 >= 128 and b2 <= 159 then
				return b2 + 64, 2
			elseif b2 >= 160 and b2 <= 191 then
				return b2 + 128, 2
			end
		elseif b1 >= 33 and b1 <= 126 then
			return b1, 1
		end

		error("character out of range")
	end

	function Tokenizer:TokenToString(token)
		local str = self.index_to_token[token]
		local out = ""
		local i = 1

		while i <= #str do
			local b, size = reverse_bytemap(str:byte(i), str:byte(i + 1))

			if b and b > 0 and b < 256 then
				out = out .. string.char(b)
			else
				out = out .. "*INVALID TOKEN*"
			end

			i = i + size
		end

		return out
	end
end

do
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

	function Tokenizer:EncodeString(str)
		local ids = {}
		local i = 1

		while i <= #str do
			local found = false

			if str:sub(i, i + 1) == "<|" then
				for _, v in ipairs(self.special_token_list) do
					if str:sub(i, i + #v - 1) == v then
						i = i + #v
						table.insert(ids, self.special_tokens[v])
						found = true

						break
					end
				end
			end

			if not found then
				table.insert(ids, self.token_to_index[bytemap(str:byte(i))])
				i = i + 1
			end
		end

		while #ids > 2 do
			local stats = {}

			for i = 1, #ids - 1 do
				local key = ids[i + 0] .. TOKEN_MERGE_SEPARATOR .. ids[i + 1]
				stats[key] = (stats[key] or 0) + 1
			end

			local minPair
			local minIndex = math.huge

			for pair, index in pairs(stats) do
				local mergeIndex = self.merges[pair] or math.huge

				if mergeIndex < minIndex then
					minIndex = mergeIndex
					minPair = pair
				end
			end

			if not self.merges[minPair] then break end

			local newIds = {}
			local i = 1

			while i <= #ids do
				local key = ids[i + 1] and ids[i] .. TOKEN_MERGE_SEPARATOR .. ids[i + 1]

				if minPair == key then
					table.insert(newIds, self.merges[minPair])
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
end

return Tokenizer