include("nnviz.jl")

name_list = [
    "split_50_big"
]

for name in name_list
    create_run_plots(name)
end