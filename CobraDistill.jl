
using Revise

using Statistics, DataFrames, Random
using Plots
using JLD

using Tiger
using JuMP, Gurobi
using Flux, CUDA


hyper = (
    widths = [1024, 512, 256, 128],
    activations = [elu, elu, elu, elu, elu],
    #loss = Flux.Losses.crossentropy,
    loss = Flux.Losses.mse,

    n_epochs = 10,
    n_samples = 100,
    batch_size = 1,
    replace_fraction = 0.0,

    optimizer = Descent,
    learning_rate = 1e-4,
    l1_regularization = 1e-5,
    l2_regularization = 1e-5,

    save_every = 3,
    test_every = 3,
    rundir = "testing"
)

# ---------------- loading Cobra model ----------------

function load_cobra(modelfile, varname; 
                    gene_ub=1000, ex_ub=100, remove_ngam=true, 
                    media_file="CDM.toml",
                    exchanges=nothing, genes=nothing)
    cobra = read_cobra(modelfile, varname)
    if remove_ngam
        cobra.lb[cobra.lb .> 0.0] .= 0.0
    end
    if !isnothing(media_file)
        set_media_bounds!(cobra, media_file)
    end
    if isnothing(exchanges)
        exchanges = get_exchange_rxns(cobra)
    end
    if isnothing(genes)
        genes = cobra.genes
    end
    cobra = extend_cobra_cnf(cobra, ub=gene_ub)
    model = build_base_model(cobra)
    binvars = variable_by_name.(model, exchanges)
    convars = variable_by_name.(model, genes)

    return model, binvars, convars
end

GENE_UB = 1000
EX_UB = 100 # from CDM.toml

model, binvars, convars = load_cobra(
    "iSMU.mat", "iSMU"; gene_ub=GENE_UB, media_file="CDM.toml", 
    exchanges=readlines("iSMU_amino_acid_exchanges.txt")
)

# ---------------- Oracle building ----------------

nbin = length(binvars)
ncon = length(convars)

function build_oracle(model, binvars, convars; bin_ub=1.0, con_ub=1.0, normalize=true, copy=true, optimizer=Gurobi.Optimizer, silent=true)
    if copy
        model, reference_map = copy_model(model)
        set_optimizer(model, optimizer)
        binvars = reference_map[binvars]
        convars = reference_map[convars]
    end

    vars = vcat(binvars, convars)
    ub = vcat(bin_ub .* ones(length(binvars)), con_ub .* ones(length(convars)))

    if silent
        set_silent(model)
    end

    if normalize
        set_upper_bound.(vars, ub)
        optimize!(model)
        max_objval = objective_value(model)
    else
        max_objval = 1.0
    end

    function run_model(X)
        n = size(X, 2)
        output = zeros(n)
        for i = 1:n
            set_upper_bound.(vars, ub .* X[:,i])
            optimize!(model)
            output[i] = objective_value(model) / max_objval
        end
        return output
    end

    return run_model
end

oracle = build_oracle(model, binvars, convars; bin_ub=EX_UB, con_ub=GENE_UB)

# ---------------- Neural Net building ----------------

ntotal = nbin + ncon
pushfirst!(hyper.widths, ntotal)
push!(hyper.widths, 1)
layers = Vector{Any}(undef, length(hyper.activations))
for i = 1:length(hyper.widths)-1
    layers[i] = Dense(hyper.widths[i], hyper.widths[i+1], hyper.activations[i])
end
nn = Chain(layers...)

# ---------------- Neural Net training ----------------

function make_sample_random(n, model, binvars, convars)
    nbin = length(binvars)
    ncon = length(convars)

    binvals = zeros(nbin, n)
    card_bin = rand(1:nbin, n)
    for i = 1:n
        binvals[rand(1:nbin, card_bin[i]),i] .= 1.0
    end

    convals = rand(ncon, n)

    return vcat(binvals, convals)
end

function update_stats!(stats, epoch, ŷ, y; test=false)
    if !(epoch in stats.epoch)
        newdf = DataFrame(
            epoch=epoch, 
            test_mean=0.0,
            test_max=0.0,
            train_mean=0.0,
            train_max=0.0
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
    mean_col[row] = mean(abs.(ŷ .- y))
    max_col[row] = maximum(abs.(ŷ .- y))
end

function make_run_dir()
    runpath = "runs/" * hyper.rundir * "/"
    epochpath = runpath * "epochs/"
    mkpath(epochpath)
    open(runpath * "hyper.txt", "w") do io
        print(io, hyper)
    end
    return runpath, epochpath
end

function save_epoch(epoch_path, epoch, stats, nn, ŷ, y)
    save(epoch_path * string(epoch) * ".jld", 
         "stats", stats, "nn", nn, "ŷ", ŷ, "y", y)
end


ps = params(nn)
opt = hyper.optimizer(hyper.learning_rate)

# training loop
l1_norm(x) = sum(abs, x)
l2_norm(x) = sum(abs2, x)
loss(X, y) = hyper.loss(nn(X), y) + hyper.l1_regularization*sum(l1_norm, ps) + hyper.l2_regularization*sum(l2_norm, ps)

n_samples = hyper.n_samples
n_epochs = hyper.n_epochs
n_replace = trunc(Int, hyper.replace_fraction * n_samples)

stats = DataFrame(
    epoch = 1:0,
    test_mean = zeros(0),
    test_max = zeros(0),
    train_mean = zeros(0),
    train_max = zeros(0)
)
run_path, epoch_path = make_run_dir()

X = make_sample_random(n_samples, model, binvars, convars)
y = oracle(X)
for epoch = 1:n_epochs
    if n_replace > 0
        Xnew = make_sample_random(n_replace, model, binvars, convars)
        ynew = oracle(Xnew)
        locs = Random.randperm(n_samples)[1:n_replace]
        X[:,locs] = Xnew
        y[locs] = ynew
    end

    println("Epoch ", epoch)
    if epoch % hyper.test_every == 0
        update_stats!(stats, epoch, nn(X), y, test=true)
    end

    data = Flux.DataLoader((X, hcat(y)'), batchsize=hyper.batch_size)
    Flux.train!(loss, ps, data, opt)

    if epoch % hyper.test_every == 0
        ŷ = nn(X)
        update_stats!(stats, epoch, ŷ, y, test=false)
    end

    if epoch % hyper.save_every == 0
        save_epoch(epoch_path, epoch, stats, nn, ŷ, y)
    end
end





