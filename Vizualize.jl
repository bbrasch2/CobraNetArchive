include("nnviz.jl")

name_list = [
    "rejection_random_70_big"
]

for name in name_list
    create_run_plots(name, plot_lr=false)
end