include("nnviz.jl")

name_list = [
    "lr_decay_optimal2"
]

for name in name_list
    create_run_plots(name)
end