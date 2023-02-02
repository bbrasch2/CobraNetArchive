using Flux

include("CobraDistill.jl")
include("nnviz.jl")

name_list = [
    "rejection_nadam9"
]

for name in name_list
    model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    nn = get_nn(name)
    y, ŷ = evaluate_nn(nn, oracle, model, binvars, convars, 10000)
    create_eval_plots(name, y, ŷ)
end