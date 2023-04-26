using Revise
using CSV, BSON
using Statistics, DataFrames, Random, LinearAlgebra
using Flux, CUDA

# Creates a NamedTuple object of hyperparameters based on input
# specifications and defaults
#
# Defaults:
#   * ELU activation functions
#   * MSE loss function
#   * NADAM optimizer
#   * No regularization
#   * Validate every 10 epochs
#   * Save network every 10 epochs
function make_hyper(widths_in, out_activation, n_epochs_in, batch_size_in, val_split_in,
    optimizer_in, learning_rate_in, decay_in, decay_start_in, l1_in, l2_in, 
    dropout_in, rundir_in, datadir_in, cobranetdir_in)
    
    hyper = (
        widths = widths_in,
        activations = [fill(elu, length(widths_in))..., out_activation], # input layer & hidden layers
        loss = Flux.Losses.mse,

        n_epochs = n_epochs_in,
        val_split = val_split_in, # proportion of data to be set aside for validation
        batch_size = batch_size_in,

        optimizer = optimizer_in,
        learning_rate = learning_rate_in,
        decay = decay_in,
        decay_start = decay_start_in,
        l1_regularization = l1_in,
        l2_regularization = l2_in,
        dropout = dropout_in,

        test_every = 10,
        save_every = 250,  # must be a multiple of test_every
        rundir = rundir_in,
        datadir = datadir_in,
        cobranetdir = cobranetdir_in,
        exch_rxns_filepath = "iSMU_amino_acid_exchanges.txt",
        genes_filepath = "iSMU_amino_acid_genes.txt",
    )
    return hyper
end

# ---------------- Stats ----------------
# This section contains helper functions to calculating test 
# and train stats. It also creates a directory for each run that 
# holds the trained model and data. The functions in nnviz.jl (which 
# don't exist yet) can be run later to create plots in the run 
# directory. The main training loop does not output anything.

function make_run_dir(hyper)
    runpath = "streptoruns/" * hyper.rundir * "/"
    epochpath = runpath * "epochs/"
    mkpath(epochpath)
    open(runpath * "hyper.txt", "w") do io
        println(io, hyper)
    end
    return runpath, epochpath
end

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

function save_epoch(epoch_path, epoch, stats, nn, X, ŷ, y, genes)
    bson(epoch_path * string(epoch) * ".bson", Dict("stats" => stats,
        "nn" => nn, "X" => X, "ŷ" => ŷ, "y" => y, "genes" => genes))
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
# The oracle has n_inputs inputs and n_outputs outputs. Using the 
# widths and activations in the tuple `hyper`, the NN structure is
#
#   Chain(
#       Dense(n_inputs,    widths[1], activations[1]),
#       Dense(widths[1],   widths[2], activations[2]), 
#       ...,
#       Dense(widths[i-1], widths[i], activations[i]),
#       ...,
#       Dense(widths[end], n_outputs, activations[end])
#   )
function make_nn(hyper, n_inputs, n_outputs)
    pushfirst!(hyper.widths, n_inputs)
    push!(hyper.widths, n_outputs)
    layers = Vector{Any}(undef, length(hyper.activations) * 2)
    for i = 1:length(hyper.widths)-1
        layers[i*2-1] = Dense(hyper.widths[i], hyper.widths[i+1], hyper.activations[i])
        layers[i*2] = Dropout(hyper.dropout)
    end
    nn = Chain(layers...)
    return nn
end

# ---------------- Neural Net training ----------------
function train_nn(hyper,nn,stats,epoch_path,epoch_skips)
    Xdata, ydata = get_data(hyper) # load datadir
    num_samples = size(Xdata, 2)
    locs = Random.randperm(num_samples)
    idx = Int(ceil(num_samples * hyper.val_split))
    Xtest = Xdata[:,locs[1:idx]]
    ytest = ydata[locs[1:idx]]
    X = Xdata[:,locs[idx+1:end]]
    y = ydata[locs[idx+1:end]]
    cobranet = get_nn(hyper) # load from cobranetdir

    ps = params(nn)
    opt = hyper.optimizer(hyper.learning_rate)

    # Using both L1 and L2 regularization, but either can be zeroed
    # out using `hyper.l*_regularization`.
    l1_norm(x) = sum(abs, x)
    l2_norm(x) = sum(abs2, x)
    loss(X, y) = hyper.loss(cobranet(vcat(X,nn(X))), y) +
        hyper.l1_regularization*sum(l1_norm, ps) + 
        hyper.l2_regularization*sum(l2_norm, ps)

    epoch_start = epoch_skips + 1
    for epoch = epoch_start:hyper.n_epochs
        println("Epoch ", epoch)
        flush(stdout)

        # Randomly shuffle sample order
        locs = Random.randperm(size(X, 2))
        X = X[:,locs]
        y = y[locs]

        data_load = Flux.DataLoader((X, hcat(y)'), batchsize=hyper.batch_size)
        Flux.train!(loss, ps, data_load, opt)

        if epoch % hyper.test_every == 0
            genes = nn(X)
            test_genes = nn(Xtest)
            ŷ = vec(cobranet(vcat(X,genes))')
            update_stats!(stats, epoch, ŷ, y, opt.eta, test=false)
            ŷtest = vec(cobranet(vcat(Xtest,test_genes))')
            update_stats!(stats, epoch, ŷtest, ytest, opt.eta, test=true)
        end

        if epoch % hyper.save_every == 0
            # notice how this relies on ŷtest, which is only computed
            # every `hyper.test_every` epochs. So `hyper.save_every`
            # needs to be a multiple of `hyper.test_every`.
            save_epoch(epoch_path, epoch, stats, nn, Xtest, ŷtest, ytest, test_genes)
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
end

# ---------------- Resuming Training ----------------
# Checks the status of a training protocol and returns the 
# most recently-updated network and an integer representing 
# the number of epochs to skip
function get_training_status(hyper)
    n_inputs = length(readlines(hyper.exch_rxns_filepath))
    n_outputs = length(readlines(hyper.genes_filepath))
    println("Getting training status of ", hyper.rundir)
    epoch_path = "streptoruns/" * hyper.rundir * "/epochs/"
    if !isdir(epoch_path)
        # Directory does not exist, so training has not started
        stats, _, epoch_path = make_stats(hyper)
        nn = make_nn(hyper, n_inputs, n_outputs)
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
        nn = make_nn(hyper, n_inputs, n_outputs)
        println("Training not started, initiating training.")
        return nn, stats, epoch_path, 0
    end
    bson = BSON.load(most_recent_saved)
    nn = bson["nn"]
    stats = bson["stats"]
    println("Training started, loading network saved at epoch: " * string(max_epoch) * ".")
    return nn, stats, epoch_path, max_epoch
end

function get_nn(hyper)
    epoch_path = "runs/" * hyper.cobranetdir * "/epochs/"
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
    epoch_path = "streptoruns/" * name * "/epochs/"
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

function get_data(hyper)
    csv_filepath = "exp_data/" * hyper.datadir * ".csv"

    # Load into dataframe
    data = DataFrame(CSV.File(csv_filepath))

    # Extract inputs
    binvars = chop.(readlines(hyper.exch_rxns_filepath), tail=5)
    AA_subset = data[!, binvars]
    binvals = Matrix{Float32}(AA_subset)'

    # Extract outputs
    fitness = data[!,"fitness_mean"]

    return binvals, fitness
end

function get_absolute_error(stats, num_rows)
    if isnothing(stats)
        return "DNE", "DNE"
    else
        train_mean = mean(stats.train_mean[max(1,end-num_rows+1):end])
        test_mean = mean(stats.test_mean[max(1,end-num_rows+1):end])
        return train_mean, test_mean
    end
end