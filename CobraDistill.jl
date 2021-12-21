
using Revise

using Statistics
using Plots

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

    if silent
        set_silent(model)
    end

    if normalize
        set_upper_bound.(binvars, bin_ub)
        set_upper_bound.(convars, con_ub)
        optimize!(model)
        max_objval = objective_value(model)
    else
        max_objval = 1.0
    end

    function run_model(binvals, convals)
        n = size(binvals, 2)
        output = zeros(n)
        for i = 1:n
            set_upper_bound.(binvars, bin_ub .* binvals[:,i])
            set_upper_bound.(convars, con_ub .* convals[:,i])
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

    return binvals, convals
end

#Xb, Xc = make_sample_random(10, model, binvars, convars)
#y = oracle(Xb, Xc)

function plot_test_train(epoch, test_mean, test_max, train_mean, train_max)
    p_train = plot(1:epoch, hcat(train_mean[1:epoch], train_max[1:epoch]), plot_title="train", ylim=(0,1), legend=false)
    p_test = plot([1:epoch], hcat(test_mean[1:epoch], test_max[1:epoch]), plot_title="test", ylim=(0,1), legend=false)
    display(plot!(p_train, p_test, plot_title=""))
end

ntotal = nbin + ncon
NEURONS = 512
nn = Chain(
    Dense(ntotal, 1024, relu),
    Dense(1024, NEURONS, relu),
    Dense(NEURONS, NEURONS, relu),
    Dense(NEURONS, NEURONS, relu),
    Dense(NEURONS, 1, relu)
) |> gpu

function mean_max_loss(nn, Xb, Xc, y)
    yhat = nn(vcat(Xb, Xc) |> gpu) |> cpu
    return mean(abs.(yhat .- y)), maximum(abs.(yhat .- y))
end

# training loop
loss(X, y) = sum((nn(X) .- y).^2)

ps = params(nn)
#opt = Descent(0.01)
opt = ADAM(0.01)

n_samples = 1000
n_epoch = 10

test_mean = zeros(n_epoch)
test_max = zeros(n_epoch)
train_mean = zeros(n_epoch)
train_max = zeros(n_epoch)

plot_test_train(1, test_mean, test_max, train_mean, train_max)

for epoch = 1:n_epoch
    Xb, Xc = make_sample_random(n_samples, model, binvars, convars)
    y = oracle(Xb, Xc)

    println("Epoch ", epoch)
    test_mean[epoch], test_max[epoch] = mean_max_loss(nn, Xb, Xc, y)

    X = vcat(Xb, Xc) |> gpu
    data = Flux.DataLoader((X, vcat(y) |> gpu))

    Flux.train!(loss, ps, data, opt)

    # for i = 1:n_samples
    #     gs = gradient(ps) do
    #         loss(X[:,i], y[i])
    #     end
    #     Flux.Optimise.update!(opt, ps, gs)
    # end

    # error reporting
    train_mean[epoch], train_max[epoch] = mean_max_loss(nn, Xb, Xc, y)
    plot_test_train(epoch, test_mean, test_max, train_mean, train_max)
end





