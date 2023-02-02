using Random
using Tiger
using JuMP, Gurobi

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
    objvars = variable_by_name.(model, "bio00001")
                                                # find label for objective function at end
                                                # then constrain at specific (random?) value 
                                                # and minimize binvars

    return model, binvars, convars, objvars
end

GENE_UB = 1000
EX_UB = 100 # from CDM.toml

model, binvars, convars, objvars = load_cobra(
    "iSMU.mat", "iSMU"; gene_ub=GENE_UB, media_file="CDM.toml", 
    exchanges=readlines("iSMU_amino_acid_exchanges.txt"),
    genes=readlines("iSMU_amino_acid_genes.txt")
)

# ---------------- Oracle building ----------------

nbin = length(binvars)
ncon = length(convars)

function build_oracle(model, binvars, convars, objvars; bin_ub=1.0, con_ub=1.0, normalize=true, copy=true, optimizer=Gurobi.Optimizer, silent=true, reverse=false)
    if copy
        model, reference_map = copy_model(model)
        set_optimizer(model, optimizer)
        binvars = reference_map[binvars]
        convars = reference_map[convars]
        objvars = reference_map[objvars]
    end

    # Generate reverse model as well
    rev_model, rev_ref_map = copy_model(model)
    set_optimizer(rev_model, optimizer)

    vars = vcat(binvars, convars)
    ub = vcat(bin_ub .* ones(length(binvars)), con_ub .* ones(length(convars)))

    if silent
        set_silent(model)
        set_silent(rev_model)
    end

    if normalize
        set_upper_bound.(vars, ub)
        optimize!(model)
        max_objval = objective_value(model)
    else
        max_objval = 1.0
    end

    function run_model(X, Y)
        if X != nothing
            n_x = size(X, 2)
        else
            n_x = 0
        end
        if Y != nothing
            n_y = length(Y)
            shuffle = true
        else
            n_y = 0
            shuffle = false
        end

        n = n_x + n_y
        x_output = zeros(size(vars, 1), n)
        y_output = zeros(n)

        for i = 1:n
            if i <= n_x
                set_upper_bound.(vars, ub .* X[:,i])
                optimize!(model)
                x_output[:,i] = X[:,i] 
                y_output[i] = objective_value(model) / max_objval
            else
                # IN PROGRESS: Test logarithmic weighting inside sum
                # IN PROGRESS: Randomize proportion of fixed/min vars
                optimal = false
                while !optimal
                    # Set fix ratio (0.0 = minimize all vars, 
                    # 1.0 = upped bound all vars at random values)
                    fix_ratio = rand(Float32)
                    min_vars = []
                    
                    #@objective(rev_model, Min, sum(weighting .* rev_ref_map[vars]))
                    for j = 1:length(vars)
                        var = vars[j]
                        if rand(Float32) < fix_ratio
                            set_upper_bound.(rev_ref_map[var], ub[j] * rand(Float32))
                        else
                            append!(min_vars, j)
                        end
                    end
                    log_weighting_factor = 0
                    weighting = 10 .^ (rand(Float32, length(min_vars)) * log_weighting_factor)
                    @objective(rev_model, Min, sum(weighting .* rev_ref_map[vars[min_vars]]))
                    fix(rev_ref_map[objvars], Y[i - n_x] * max_objval, force=true)
                    optimize!(rev_model)
                    # Validate solution before continuing
                    optimal = (termination_status(rev_model) == MOI.OPTIMAL)
                end
                x_output[:,i] = value.(rev_ref_map[vars])
                y_output[i] = Y[i - n_x]
            end
        end
        
        # Randomly shuffle arrays
        if shuffle
            shuf = Random.shuffle(range(start=1, stop=n, step=1))
            x_output = x_output[:,shuf]
            y_output = y_output[shuf]
        end
        return x_output, y_output
    end
    return run_model
end

oracle = build_oracle(model, binvars, convars, objvars; bin_ub=EX_UB, con_ub=GENE_UB)

ntotal = nbin + ncon
