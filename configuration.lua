local function Configuration(context_length, metadata, token_size)
    local model_context_length = metadata["llama.context_length"]
    local dim = metadata["llama.embedding_length"]
    local hiddenDim = metadata["llama.feed_forward_length"]
    local numberOfLayers = metadata["llama.block_count"]
    local numberOfHeads = metadata["llama.attention.head_count"]
    local numberOfKeyValueHeads = metadata["llama.attention.head_count_kv"] and
        metadata["llama.attention.head_count_kv"] or
        metadata["llama.attention.head_count"]
    local vocabularySize = token_size
    local sharedWeights = false
    local rmsNormEps = metadata["llama.attention.layer_norm_rms_epsilon"] or 1e-5
    local ropeTheta = metadata["llama.rope.freq_base"] or 10000
    local headSize = dim / numberOfHeads

    local ropeFreqsReal, ropeFreqsImag = (function()
        assert(headSize % 2 == 0)
        local cr = {}
        local ci = {}
        local n = 1

        for pos = 0, context_length - 1 do
            for i = 0, headSize - 1, 2 do
                local freq = 1.0 / (ropeTheta ^ (i / headSize))
                local val = pos * freq
                cr[n] = math.cos(val)
                ci[n] = math.sin(val)
                n = n + 1
            end
        end

        n = n - 1
        assert(context_length * (headSize / 2) == n)
        return cr, ci
    end)()

    return {
        tokenizer = tokenizer,
        context_length = context_length,
        model_context_length = model_context_length,
        dim = dim,
        hiddenDim = hiddenDim,
        numberOfLayers = numberOfLayers,
        numberOfHeads = numberOfHeads,
        numberOfKeyValueHeads = numberOfKeyValueHeads,
        vocabularySize = vocabularySize,
        contextLength = context_length,
        sharedWeights = sharedWeights,
        rmsNormEps = rmsNormEps,
        ropeTheta = ropeTheta,
        headSize = headSize,
        ropeFreqsReal = ropeFreqsReal,
        ropeFreqsImag = ropeFreqsImag,
    }
end

return Configuration