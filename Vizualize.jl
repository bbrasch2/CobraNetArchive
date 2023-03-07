include("nnviz.jl")

name_list = [
    "new_data_test",
    "new_data_test2"
]

for name in name_list
    create_run_plots(name, plot_lr=false)
end