using Random

include("cobra_model.jl")
include("caching.jl")

function make_sample_random(n, oracle, model, binvars, convars)
    nbin = length(binvars)
    ncon = length(convars)

    # IN PROGRESS: test sampling binvals with continuous values
    binvals = rand(Float32, nbin, n) # zeros(Float32, nbin, n)
    #card_bin = rand(1:nbin, n)
    #for i = 1:n
    #    binvals[rand(1:nbin, card_bin[i]),i] .= 1.0
    #end

    convals = rand(Float32, ncon, n)

    X = vcat(binvals, convals)
    X, Y = oracle(X, nothing)

    return X, convert.(Float32, y)
end

function make_space_filler(n, oracle, model, binvars, convars, n_bins=100, binary=true)
    nbin = length(binvars)
    ncon = length(convars)

    binvals = zeros(Float32, nbin, n)
    convals = zeros(Float32, ncon, n)

    counter = 1
    bin_cap = ceil(n / n_bins)
    bin_counts = Int.(zeros(n_bins))

    while counter <= n
        convals_temp = rand(Float32, ncon, n)
        if binary
            binvals_temp = zeros(Float32, nbin, n)
            card_bin = rand(1:nbin, n)
            for i = 1:n
                binvals_temp[rand(1:nbin, card_bin[i]),i] .= 1.0
            end
        else
            binvals_temp = rand(Float32, nbin, n)
        end

        X = vcat(binvals_temp, convals_temp)
        X, Y = oracle(X, nothing)
        Y = convert.(Float32, Y)

        for i = 1:n
            idx = Int(ceil(Y[i] * n_bins))
            if idx == 0
                idx += 1
            end

            if bin_counts[idx] < bin_cap
                convals[:,counter] = convals_temp[:,i]
                binvals[:,counter] = binvals_temp[:,i]
                bin_counts[idx] += 1
                counter += 1
            end

            if counter > n
                break
            end
        end
    end

    X = vcat(binvals, convals)
    X = X[:,Random.shuffle(1:end)]
    X, Y = oracle(X, nothing)
    Y = convert.(Float32, Y)
    return X, Y
end

function generate_space_filled(n, oracle, model)
    Y = rand(Float32, n)
    X, Y = oracle(nothing, Y)
    X = convert.(Float32, X)
    return X, Y
end

function generate_mixed_data(n_x, n_y, oracle, model, binvars, convars)
    nbin = length(binvars)
    ncon = length(convars)

    binvals = zeros(Float32, nbin, n_x)
    card_bin = rand(1:nbin, n_x)
    for i = 1:n_x
        binvals[rand(1:nbin, card_bin[i]),i] .= 1.0
    end
    convals = rand(Float32, ncon, n_x)
    X = vcat(binvals, convals)
    Y = rand(Float32, n_y)
    X, Y = oracle(X, Y)
    return X, Y
end

#sampler(n) = make_space_filler(n, oracle, model, binvars, convars, 100, true)

#cache_training_data(10000, 1000, sampler, "cache/rejection")
mix_cache("cache/rejection", "cache/random10M", "cache/rejection_random_70", 0.7, 550)