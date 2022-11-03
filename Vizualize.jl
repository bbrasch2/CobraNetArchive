include("nnviz.jl")

name_list = [
    "testdecay19"
]

for name in name_list
    create_run_plots(name)
end