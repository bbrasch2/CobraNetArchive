
using Revise

using Statistics, DataFrames
using Plots
using JLD

using Tiger
using JuMP, Gurobi
using Flux, CUDA

GENE_UB = 1000
EX_UB = 100 # from CDM.toml

cobra = read_cobra("iSMU.mat", "iSMU")
cobra.lb[cobra.lb .> 0.0] .= 0.0  # remove NGAM
cobra = extend_cobra_cnf(cobra, ub=GENE_UB)
set_media_bounds!(cobra, "CDM.toml")

amino_acid_exchanges = [
"ala_exch", 
"arg_exch", 
"asp_exch", 
"asn_exch", 
"cys_exch", 
"glu_exch", 
"gln_exch", 
"gly_exch", 
"his_exch", 
"ile_exch", 
"leu_exch", 
"lys_exch", 
"met_exch", 
"phe_exch", 
"pro_exch", 
"ser_exch", 
"thr_exch", 
"trp_exch", 
"tyr_exch", 
"val_exch" ];

model = build_base_model(cobra)
binvars = variable_by_name.(model, amino_acid_exchanges)
convars = variable_by_name.(model, cobra.genes)

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

function plot_test_train(epoch, test_mean, test_max, train_mean, train_max)
    p_train = plot(1:epoch, hcat(train_mean[1:epoch], train_max[1:epoch]), plot_title="train", ylim=(0,1), legend=false)
    p_test = plot([1:epoch], hcat(test_mean[1:epoch], test_max[1:epoch]), plot_title="test", ylim=(0,1), legend=false)
    display(plot!(p_train, p_test, plot_title=""))
end

hyper = (
    n_epochs = 100,
    n_samples = 10000,
    batch_size = 32,

    optimizer = ADAM,
    learning_rate = 1e-10,

    loss = Flux.Losses.crossentropy,

    save_every = 100,
    save_prefix = "train10_"
)

ntotal = nbin + ncon
NEURONS = 512
nn = Chain(
    Dense(ntotal, 1024, elu),
    #Dense(1024, 512, elu),
    Dense(1024, 256, elu),
    Dense(256, 128, elu),
    Dense(128, 1, σ)
)

function update_stats!(stats, epoch, ŷ, y; test=false)
    if test
        mean_col = stats.test_mean
        max_col = stats.test_max
    else
        mean_col = stats.train_mean
        max_col = stats.train_max
    end

    mean_col[epoch] = mean(abs.(ŷ .- y))
    max_col[epoch] = maximum(abs.(ŷ .- y))
end

# training loop
loss(X, y) = hyper.loss(nn(X), y)

ps = params(nn)
opt = hyper.optimizer(hyper.learning_rate)

n_samples = hyper.n_samples
n_epochs = hyper.n_epochs

stats = DataFrame(
    epoch = 1:n_epochs,
    test_mean = zeros(n_epochs),
    test_max = zeros(n_epochs),
    train_mean = zeros(n_epochs),
    train_max = zeros(n_epochs)
)

X = make_sample_random(n_samples, model, binvars, convars)
y = oracle(X)
for epoch = 1:n_epochs
    println("Epoch ", epoch)
    update_stats!(stats, epoch, nn(X), y, test=true)

    data = Flux.DataLoader((X, vcat(y)), batchsize=hyper.batch_size)
    Flux.train!(loss, ps, data, opt)

    ŷ = nn(X)
    update_stats!(stats, epoch, ŷ, y, test=false)

    if epoch % hyper.save_every == 0
        save(hyper.save_prefix * string(epoch) * ".jld",
            "stats", stats, "nn", nn, "ŷ", ŷ, "y", y)
    end

end





