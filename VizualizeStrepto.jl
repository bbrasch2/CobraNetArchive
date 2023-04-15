include("streptoviz.jl")

name_list = [
    "dropout"
]

for name in name_list
    create_run_plots(name)
end