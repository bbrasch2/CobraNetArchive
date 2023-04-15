using Revise

using Statistics, DataFrames, Random
using Plots
using BSON

using Flux, CUDA

include("caching.jl")

function make_hyper(widths_in, activations_in, n_epochs_in, batch_size_in,
                    optimizer_in, learning_rate_in, decay_in, decay_start_in, 
                    rundir_in, cachedir_in)
    hyper = (
        widths = widths_in,
        activations = activations_in,
        loss = Flux.Losses.mse,

        n_epochs = n_epochs_in,
        n_samples = 10000,
        n_test = 10000,
        batch_size = batch_size_in,
        replace_fraction = 0.1,

        #optimizer = Flux.ADAM,
        #optimizer = Flux.ADADelta,
        #optimizer = Flux.NADAM,
        #optimizer = Flux.AdaMax,
        optimizer = optimizer_in,
        learning_rate = learning_rate_in,
        decay = decay_in,
        decay_start = decay_start_in,
        l1_regularization = 0.0,
        l2_regularization = 0.0,

        test_every = 10,
        save_every = 1000,  # must be a multiple of test_every
        rundir = rundir_in,

        cached = true,
        cachedir = "cache/" * cachedir_in * "/",
        skip_completed_cache = true
    )
    return hyper
end

# ---------------- loading Cobra model & oracle ----------------
# model, binvars, convars
# oracle
# nbin, ncon, ntotal
function load_cobra_model()
    include("cobra_model.jl")
    return model, binvars, convars, oracle, nbin, ncon, ntotal
end

# ---------------- Sampling ----------------

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

    Y = oracle(X)

    return X, convert.(Float32, Y)
end

# ---------------- Stats ----------------
# This section contains helper functions to calculating test 
# and train stats. It also creates a directory for each run that 
# holds the trained model and data. The functions in nnviz.jl 
# can be run later to create plots in the run directory. The main 
# training loop does not output anything.

function update_stats!(stats, epoch, ŷ, y, lr; test=false)
    if !(epoch in stats.epoch)
        newdf = DataFrame(
            epoch=epoch, 
            test_mean=0.0,
            test_max=0.0,
            train_mean=0.0,
            train_max=0.0,
            learning_rate=lr
        )
        append!(stats, newdf)
    end

    if test
        mean_col = stats.test_mean
        max_col = stats.test_max
    else
        mean_col = stats.train_mean
        max_col = stats.train_max
    end

    row = findfirst(stats.epoch .== epoch)
    mean_col[row] = mean(abs.(ŷ - y))
    max_col[row] = maximum(abs.(ŷ - y))
end

function make_run_dir(hyper)
    runpath = "runs/" * hyper.rundir * "/"
    epochpath = runpath * "epochs/"
    mkpath(epochpath)
    open(runpath * "hyper.txt", "w") do io
        println(io, hyper)
    end
    return runpath, epochpath
end

function save_epoch(epoch_path, epoch, stats, nn, ŷ, y)
    bson(epoch_path * string(epoch) * ".bson", Dict("stats" => stats,
        "nn" => nn, "ŷ" => ŷ, "y" => y))
end

# Now we can create the stats DF and the run directory for this 
# training run.
function make_stats(hyper)
    stats = DataFrame(
        epoch = 1:0,
        test_mean = zeros(0),
        test_max = zeros(0),
        train_mean = zeros(0),
        train_max = zeros(0),
        learning_rate = zeros(0)
    )
    run_path, epoch_path = make_run_dir(hyper)
    return stats, run_path, epoch_path
end

# ---------------- Neural Net building ----------------
# The oracle has ntotal inputs and 1 output. Using the widths
# and activations in the tuple `hyper`, the NN structure is
#
#   Chain(
#       Dense(ntotal,      widths[1], activations[1]),
#       Dense(widths[1],   widths[2], activations[2]), 
#       ...,
#       Dense(widths[i-1], widths[i], activations[i]),
#       ...,
#       Dense(widths[end], 1,         activations[end])
#   )
function make_nn(hyper,ntotal)
    pushfirst!(hyper.widths, ntotal)
    push!(hyper.widths, 1)
    layers = Vector{Any}(undef, length(hyper.activations))
    for i = 1:length(hyper.widths)-1
        layers[i] = Dense(hyper.widths[i], hyper.widths[i+1], hyper.activations[i])
    end
    nn = Chain(layers...)
    return nn
end

# ---------------- Neural Net training ----------------
function train_nn(hyper,nn,oracle,model,binvars,convars,stats,epoch_path,epoch_skips)
    n_samples = hyper.n_samples
    n_replace = trunc(Int, hyper.replace_fraction * n_samples)
    n_test = hyper.n_test
    cachedir = hyper.cachedir

    # Before training we need to generate test data (Xtest, ytest)
    # and the first epoch of training data (X, y). During each epoch,
    # n_replace of the training entries are randomly replaced with 
    # new training data. The training loop calls next_batch() to 
    # generate the n_replace new observations.
    # 
    # If we're using a cache, we first read the test and inital 
    # training batches. Then we define start a thread to read batches
    # from file and wrap the channel with next_batch().
    #
    # If no cache is used, we generate random test and train data
    # and wrap the random sampler with next_batch().
    if hyper.cached
        Xtest, ytest = get_batch(cachedir, n_test)
        X, y = get_batch(cachedir, n_samples; skip=n_test)
        batch_channel = Channel(1)
        errormonitor(@async serve_batches(batch_channel, cachedir, n_replace, hyper.n_epochs; skip=n_test+n_samples+epoch_skips))
        next_batch = () -> take!(batch_channel)
    else
        get_samples = (n) -> make_sample_random(n, oracle, model, binvars, convars)
        Xtest, ytest = get_samples(n_test)
        X, y = get_samples(n_samples)
        next_batch = () -> get_samples(n_replace)
    end

    ps = params(nn)
    opt = hyper.optimizer(hyper.learning_rate)
    #opt = hyper.optimizer(hyper.learning_rate...)
    #opt = hyper.optimizer()
    #opt = Flux.Optimise.Optimiser(hyper.optimizer(), ExpDecay(1, hyper.learning_rate, 1, 0))

    # Using both L1 and L2 regularization, but either can be zeroed
    # out using `hyper.l*_regularization`.
    l1_norm(x) = sum(abs, x)
    l2_norm(x) = sum(abs2, x)
    loss(X, y) = hyper.loss(nn(X), y) + hyper.l1_regularization*sum(l1_norm, ps) + hyper.l2_regularization*sum(l2_norm, ps)

    if hyper.skip_completed_cache
        for skip in 1:epoch_skips
            if n_replace > 0
                # insert new samples randomly in the training data
                locs = Random.randperm(hyper.n_samples)[1:n_replace]
                X[:,locs], y[locs] = next_batch()
            end
            
            # Decay learning rate
            if hyper.decay < 1 && hyper.decay_start < epoch
                opt.eta = hyper.decay * opt.eta
            end
        end
    end
    epoch_start = epoch_skips + 1
    
    for epoch = epoch_start:hyper.n_epochs
        println("Epoch ", epoch)
        flush(stdout)

        if n_replace > 0
            # insert new samples randomly in the training data
            locs = Random.randperm(hyper.n_samples)[1:n_replace]
            X[:,locs], y[locs] = next_batch()
        end

        data = Flux.DataLoader((X, hcat(y)'), batchsize=hyper.batch_size)
        Flux.train!(loss, ps, data, opt)

        if epoch % hyper.test_every == 0
            ŷ = vec(nn(X)')
            update_stats!(stats, epoch, ŷ, y, opt.eta, test=false)
            ŷtest = vec(nn(Xtest)')
            update_stats!(stats, epoch, ŷtest, ytest, opt.eta, test=true)
        end

        if epoch % hyper.save_every == 0
            # notice how this relies on ŷtest, which is only computed
            # every `hyper.test_every` epochs. So `hyper.save_every`
            # needs to be a multiple of `hyper.test_every`.
            save_epoch(epoch_path, epoch, stats, nn, ŷtest, ytest)
        end

        # Decay learning rate
        if hyper.decay < 1 && hyper.decay_start < epoch
            opt.eta = hyper.decay * opt.eta
        end

        # End training early if any NaN values found while saving
        if epoch % hyper.save_every == 0 && any(isnan.(ŷtest))
            println("NaN values found in ŷtest, ending training early.")
            flush(stdout)
            break
        end
    end

    if hyper.cached
        close(batch_channel)
        println("Closed channel")
        flush(stdout)
    end
end

# ---------------- Resuming Training ----------------
# Checks the status of a training protocol and returns the 
# most recently-updated network and an integer representing 
# the number of epochs to skip
function get_training_status(hyper,ntotal)
    println("Getting training status of ", hyper.rundir)
    epoch_path = "runs/" * hyper.rundir * "/epochs/"
    if !isdir(epoch_path)
        # Directory does not exist, so training has not started
        stats, _, epoch_path = make_stats(hyper)
        nn = make_nn(hyper,ntotal)
        println("Training not started, initiating training.")
        return nn, stats, epoch_path, 0
    end
    max_epoch = 0
    most_recent_saved = ""
    for bsonfile in readdir(epoch_path)
        filename = epoch_path * bsonfile
        epoch = parse(Int, splitext(splitdir(filename)[2])[1])
        if epoch > max_epoch
            max_epoch = epoch
            most_recent_saved = filename
        end
    end
    if max_epoch == 0
        # Output files do not exist, so training has not started
        stats, _, epoch_path = make_stats(hyper)
        nn = make_nn(hyper,ntotal)
        println("Training not started, initiating training.")
        return nn, stats, epoch_path, 0
    end
    bson = BSON.load(most_recent_saved)
    nn = bson["nn"]
    stats = bson["stats"]
    #_, epoch_path = make_run_dir(hyper)
    epoch_path = "runs/" * hyper.rundir * "/epochs/"
    println("Training started, loading network saved at epoch: " * string(max_epoch) * ".")
    return nn, stats, epoch_path, max_epoch
end

function get_nn(name)
    epoch_path = "runs/" * name * "/epochs/"
    max_epoch = 0
    most_recent_saved = ""
    for bsonfile in readdir(epoch_path)
        filename = epoch_path * bsonfile
        epoch = parse(Int, splitext(splitdir(filename)[2])[1])
        if epoch > max_epoch
            max_epoch = epoch
            most_recent_saved = filename
        end
    end
    bson = BSON.load(most_recent_saved)
    nn = bson["nn"]
    return nn
end

function get_stats(name)
    epoch_path = "runs/" * name * "/epochs/"
    max_epoch = 0
    most_recent_saved = ""
    
    if isdir(epoch_path)
        for bsonfile in readdir(epoch_path)
            filename = epoch_path * bsonfile
            epoch = parse(Int, splitext(splitdir(filename)[2])[1])
            if epoch > max_epoch
                max_epoch = epoch
                most_recent_saved = filename
            end
        end
    else
        return nothing
    end

    if isfile(most_recent_saved)
        bson = BSON.load(most_recent_saved)
        stats = bson["stats"]
        return stats
    else
        return nothing
    end
end

# ---------------- Evaluating Network ----------------
function evaluate_nn(nn, oracle, model, binvars, convars, n_samples)
    X, y = make_sample_random(n_samples, oracle, model, binvars, convars)
    ŷ = nn(X)
    return y, ŷ
end

function evaluate_nn_cache(nn, n_samples, cachedir)
    X, y = get_batch(cachedir, n_samples)
    ŷ = nn(X)
    return vec(y), vec(ŷ)
end

function get_absolute_error(stats, num_rows)
    if isnothing(stats)
        return "DNE"
    else
        test_mean = stats.test_mean[max(1,end-num_rows+1):end]
        return mean(test_mean)
    end
end