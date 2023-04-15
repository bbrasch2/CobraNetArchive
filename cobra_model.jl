using Tiger
using JuMP, Gurobi

# ---------------- loading Cobra model ----------------

function load_cobra(modelfile, varname; 
                    gene_ub=1000, ex_ub=100, remove_ngam=true, 
                    media_file="CDM.toml",
                    exchanges=nothing, genes=nothing)
    cobra = read_cobra(modelfile, varname)
    cobra = extend_cobra_cnf(cobra, ub=gene_ub)
    if remove_ngam
        cobra.lb[cobra.lb .> 0.0] .= 0.0
    end
    if isnothing(exchanges)
        exchanges = get_exchange_rxns(cobra)
    end
    if isnothing(genes)
        genes = cobra.genes
    end
    model = build_base_model(cobra)

    if !isnothing(media_file)
        set_media_bounds!(model, media_file)
    end

    binvars = variable_by_name.(model, exchanges)
    convars = variable_by_name.(model, genes)
    objvars = variable_by_name.(model, "bio00001")

    return model, binvars, convars, objvars
end

GENE_UB = 1000
EX_UB = 100 # from CDM.toml

model, binvars, convars, objvars = load_cobra(
    "iSMU_rescaled.mat", "rescaled_model"; gene_ub=GENE_UB, media_file="CDM.toml", 
    exchanges=readlines("iSMU_amino_acid_exchanges.txt"),
    genes=readlines("iSMU_amino_acid_genes.txt")
)
#model, binvars, convars, objvars = load_cobra(
#    "iSSA.mat", "iSSA"; gene_ub=GENE_UB, media_file="CDM.toml", 
#    exchanges=readlines("iSSA_amino_acid_exchanges.txt"), 
#    genes=nothing
#)

# ---------------- Oracle building ----------------

nbin = length(binvars)
ncon = length(convars)

function build_oracle(model, binvars, convars, objvars; bin_ub=1.0, con_ub=1.0, 
    normalize=true, copy=true, optimizer=Gurobi.Optimizer, silent=true)
    
    if copy
        model, reference_map = copy_model(model)
        set_optimizer(model, optimizer)
        binvars = reference_map[binvars]
        convars = reference_map[convars]
        if objvars != nothing
            objvars = reference_map[objvars]
        end
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
        num_samples = size(X, 2)
        output = zeros(num_samples)

        for i = 1:num_samples
            set_upper_bound.(vars, ub .* X[:,i])
            optimize!(model)
            output[i] = objective_value(model) / max_objval
        end
        return output
    end

    return run_model
end

oracle = build_oracle(model, binvars, convars, objvars; bin_ub=EX_UB, con_ub=GENE_UB)

ntotal = nbin + ncon
