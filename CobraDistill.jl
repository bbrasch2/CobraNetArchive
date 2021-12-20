
using Revise

using Statistics

using Tiger
using JuMP, Gurobi
using Flux

GENE_UB = 1000
EX_UB = 100 # from CDM.toml

cobra = read_cobra("/Users/jensen/Dropbox/repos/COBRA_models/iSMU.mat", "iSMU")
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

Xb, Xc = make_sample_random(10, model, binvars, convars)
y = oracle(Xb, Xc)


ntotal = nbin + ncon
NEURONS = 512
nn = Chain(
    Dense(ntotal, 1024, relu),
    Dense(1024, NEURONS, relu),
    Dense(NEURONS, NEURONS, relu),
    Dense(NEURONS, NEURONS, relu),
    Dense(NEURONS, 1, relu)
)

function mean_max_loss(nn, Xb, Xc, y)
    yhat = nn(vcat(Xb, Xc))
    return mean(abs.(yhat .- y)), maximum(abs.(yhat .- y))
end

# training loop
loss(X, y) = sum((nn(X) .- y).^2)

ps = params(nn)
opt = Descent(0.001)

n_samples = 1000
n_epoch = 10
for epoch = 1:n_epoch
    Xb, Xc = make_sample_random(n_samples, model, binvars, convars)
    y = oracle(Xb, Xc)
    X = vcat(Xb, Xc)

    println("Epoch ", epoch)
    println("    Test: ", mean_max_loss(nn, Xb, Xc, y))

    for i = 1:n_samples
        gs = gradient(ps) do
            loss(X[:,i], y[i])
        end
        Flux.Optimise.update!(opt, ps, gs)
    end

    # error reporting
    println("   Train: ", mean_max_loss(nn, Xb, Xc, y))
end




