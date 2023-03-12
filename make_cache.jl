using Random

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
    Y = convert.(Float32, oracle(X))

    return X, Y
end

function make_sample_from_AA(oracle, model, binvals, convars)
    ncon = length(convars)

    # Open up all genes
    convals = ones(Float32, ncon, size(binvals, 2))

    X = vcat(binvals, convals)
    Y = convert.(Float32, oracle(X))
    return X, Y
end

function make_space_filler(n, oracle, model, binvars, convars, n_bins=100, binary=true)
    nbin = length(binvars)
    ncon = length(convars)

    binvals = zeros(Float32, nbin, n)
    convals = zeros(Float32, ncon, n)

    counter = 1
    bin_cap = ceil(n / n_bins)
    bin_counts = Int.(zeros(n_bins))

    # Generate data until each bin is filled
    while counter <= n
        # Generate binvals & convals for n random samples
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
        Y = convert.(Float32, oracle(X))

        # Loop through the generated sample points
        for i = 1:n
            # Get bin index
            idx = Int(ceil(Y[i] * n_bins))

            # Avoid idx of 0 or n_bins+1
            if idx <= 0
                idx = 1
            elseif idx > n_bins
                idx = n_bins
            end

            # Save sample if bin still has room
            if bin_counts[idx] < bin_cap
                convals[:,counter] = convals_temp[:,i]
                binvals[:,counter] = binvals_temp[:,i]
                bin_counts[idx] += 1
                counter += 1
            end

            # Exit early if all bins are full
            if counter > n
                break
            end
        end
    end

    # Shuffle sample order before returning
    X = vcat(binvals, convals)
    X = X[:,Random.shuffle(1:end)]
    Y = convert.(Float32, oracle(X))
    return X, Y
end

cachedir = "cache/spacefill"

# Make cache of random samples
#sampler(n) = make_space_filler(n, oracle, model, binvars, convars, 10)
#cache_training_data(10000, 1000, sampler, cachedir)

# Make cache from AAs csv file
sampler(binvals) = make_sample_from_AA(oracle, model, binvals, convars)
AAs_from_csv("exp_data/SSA_aerobic_experimental_data.csv", "iSMU_amino_acid_exchanges.txt", 
    sampler, "cache/exp_data/exp_data.jld")

# Mix two cache dirs
#mix_cache("cache/rejection", "cache/random10M", "cache/rejection_random_70", 0.7, 550)