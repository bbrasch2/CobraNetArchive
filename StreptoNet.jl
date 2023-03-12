using Revise
using Statistics, DataFrames, Random
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
function make_hyper(widths_in, n_epochs_in, batch_size_in, val_split_in
    learning_rate_in, rundir_in, datadir_in, cobranetdir_in)
    
    hyper = (
        widths = widths_in,
        activations = fill(elu, length(widths) + 1) # input layer & hidden layers
        loss = Flux.Losses.mse,

        n_epochs = n_epochs_in,
        val_split = val_split_in # proportion of data to be set aside for validation
        batch_size = batch_size_in

        optimizer = Flux.NADAM,
        learning_rate = learning_rate_in,
        l1_regularization = 0.0,
        l2_regularization = 0.0,

        test_every = 10,
        save_every = 10,  # must be a multiple of test_every
        rundir = rundir_in,
        datadir = datadir_in

        cobranetdir = cobranetdir_in
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
    runpath = "runs/" * hyper.rundir * "/"
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
    layers = Vector{Any}(undef, length(hyper.activations))
    for i = 1:length(hyper.widths)-1
        layers[i] = Dense(hyper.widths[i], hyper.widths[i+1], hyper.activations[i])
    end
    nn = Chain(layers...)
    return nn
end

# ---------------- Neural Net training ----------------
function train_nn(hyper,nn,stats,datadir,cobranetdir,epoch_path,epoch_skips)
    # TODO
    data = nothing # load datadir
    Xtest = nothing # randomly select hyper.val_split of data
    ytest = nothing
    X = nothing# remaining data
    y = nothing
    cobranet = nothing # load from cobranetdir

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
        locs = Random.randperm(size(train_data, 2))
        X = X[:,locs]
        y = y[locs]

        data_load = Flux.DataLoader((X, hcat(y)'), batchsize=hyper.batch_size)
        Flux.train!(loss, ps, data_load, opt)
    end
end