using Flux

include("CobraDistill.jl")
include("nnviz.jl")

name_list = [
    "NAdam_opt_1",
    "NAdam_opt_2",
    "NAdam_opt_3",
    "NAdam_opt_4",
    "NAdam_opt_5",
    "NAdam_opt_6",
    "NAdam_opt_7",
    "NAdam_opt_8",
    "NAdam_opt_9",
    "NAdam_opt_10",
    "NAdam_opt_11",
    "NAdam_opt_12",
    "NAdam_opt_13",
    "NAdam_opt_14",
    "NAdam_opt_15",
    "NAdam_opt_16",
    "NAdam_opt_17",
    "NAdam_opt_18",
    "NAdam_opt_19",
    "NAdam_opt_20",
    "NAdam_opt_21",
    "NAdam_opt_22",
    "NAdam_opt_23",
    "NAdam_opt_24",
    "NAdam_opt_25",
    "NAdam_opt_26",
    "NAdam_opt_27",
    "NAdam_opt_28",
    "NAdam_opt_29",
    "NAdam_opt_30",
    "NAdam_opt_31",
    "NAdam_opt_32",
    "NAdam_opt_33",
    "NAdam_opt_34",
    "NAdam_opt_35",
    "NAdam_opt_36",
    "NAdam_opt_37",
    "NAdam_opt_38",
    "NAdam_opt_39",
    "NAdam_opt_40",
    "NAdam_opt_41",
    "NAdam_opt_42",
    "NAdam_opt_43",
    "NAdam_opt_44",
    "NAdam_opt_45",
    "NAdam_opt_46",
    "NAdam_opt_47",
    "NAdam_opt_48",
    "NAdam_opt_49",
    "NAdam_opt_50",
    "NAdam_opt_51",
    "NAdam_opt_52",
    "NAdam_opt_53",
    "NAdam_opt_54",
]

for name in name_list
    # Plot with validation data
    #model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    #nn = get_nn(name)
    #y, ŷ = evaluate_nn(nn, oracle, model, binvars, convars, 10000)
    #create_eval_plots(name, y, ŷ)

    # Calculate mean absolute error
    stats = get_stats(name)
    println(get_absolute_error(stats, 10))
end