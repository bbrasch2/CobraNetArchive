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

    return X, convert.(Float32, oracle(X))
end

function make_space_filler(n, oracle, model, binvars, convars)
    nbin = length(binvars)
    ncon = length(convars)

    binvals = zeros(Float32, nbin, n)
    card_bin = rand(1:nbin, n)
    for i = 1:n
        binvals[rand(1:nbin, card_bin[i]),i] .= 1.0
    end
    convals = rand(Float32, ncon, n) * 0
    filled = false
    counter = 1
    counts = [0,0,0]
    while counter <= n
        overcoverage = 1
        convals_temp = rand(Float32, ncon, n * overcoverage)
        binvals_temp = deepcopy(binvals)

        #for i = 1:(overcoverage-1)
        #    binvals_temp = hcat(binvals_temp, binvals)
        #end

        X = vcat(binvals_temp, convals_temp)
        Y = convert.(Float32, oracle(X))

        for i = 1:n #(n * overcoverage)
            if Y[i] < 0.05 && counts[1] < ceil(n/length(counts))
                convals[:,counter] = convals_temp[:,i]
                binvals[:,counter] = binvals_temp[:,i]
                counts[1] += 1
                counter += 1
            elseif Y[i] > 0.75 && counts[3] < ceil(n/length(counts))
                convals[:,counter] = convals_temp[:,i]
                binvals[:,counter] = binvals_temp[:,i]
                counts[3] += 1
                counter += 1
            elseif Y[i] >= 0.05 && Y[i] < 0.75 && counts[2] <= ceil(n/length(counts))
                convals[:,counter] = convals_temp[:,i]
                binvals[:,counter] = binvals_temp[:,i]
                counts[2] += 1
                counter += 1
            end
            if counter > n
                break
            end
        end
        #println("iteration: ", string(counter))
        #println(counts)
        #flush(stdout)
    end

    X = vcat(binvals, convals)
    X = X[:,Random.shuffle(1:end)]
    Y = convert.(Float32, oracle(X))
    return X, Y
end

function generate_space_filled(n, oracle, model)
    Y = rand(Float32, n)
    X = convert.(Float32, oracle(Y))
    return X, Y
end

#sampler(n) = make_space_filler(n, oracle, model, binvars, convars)

sampler(n) = generate_space_filled(n, oracle, model)
cache_training_data(10000, 1000, sampler, "cache/space_filled")