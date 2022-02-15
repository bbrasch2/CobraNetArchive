
include("cobra_model.jl")
include("caching.jl")

function make_sample_random(n, oracle, model, binvars, convars)
    nbin = length(binvars)
    ncon = length(convars)

    binvals = zeros(Float32, nbin, n)
    card_bin = rand(1:nbin, n)
    for i = 1:n
        binvals[rand(1:nbin, card_bin[i]),i] .= 1.0
    end

    convals = rand(Float32, ncon, n)

    X = vcat(binvals, convals)

    return X, convert.(Float32, oracle(X))
end

sampler(n) = make_sample_random(n, oracle, model, binvars, convars)
cache_training_data(10000, 1000, sampler, "cache/random10M")