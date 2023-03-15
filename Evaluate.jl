using Flux

include("CobraDistill.jl")
include("nnviz.jl")

cachedir = "cache/exp_data/"
n_samples = 3696

name_list = [
    "NAdam_CCD_1_1",
    "NAdam_CCD_2_1",
    "NAdam_CCD_3_1",
    "NAdam_CCD_4_1",
    "NAdam_CCD_5_1",
    "NAdam_CCD_6_1",
    "NAdam_CCD_7_1",
    "NAdam_CCD_8_1",
    "NAdam_CCD_9_1",
    "NAdam_CCD_10_1",
    "NAdam_CCD_11_1",
    "NAdam_CCD_12_1",
    "NAdam_CCD_13_1",
    "NAdam_CCD_14_1",
    "NAdam_CCD_15_1",
    "NAdam_CCD_16_1",
    "NAdam_CCD_17_1",
    "NAdam_CCD_18_1",
    "NAdam_CCD_19_1",
    "NAdam_CCD_20_1",
    "NAdam_CCD_21_1",
    "NAdam_CCD_22_1",
    "NAdam_CCD_23_1",
    "NAdam_CCD_24_1",
    "NAdam_CCD_25_1",
    "NAdam_CCD_26_1",
    "NAdam_CCD_27_1",
    "NAdam_CCD_28_1",
    "NAdam_CCD_29_1",
    "NAdam_CCD_30_1",
    "NAdam_CCD_31_1",
    "NAdam_CCD_32_1",
    "NAdam_CCD_33_1",
    "NAdam_CCD_34_1",
    "NAdam_CCD_35_1",
    "NAdam_CCD_36_1",
    "NAdam_CCD_37_1",
    "NAdam_CCD_38_1",
    "NAdam_CCD_39_1",
    "NAdam_CCD_40_1",
    "NAdam_CCD_41_1",
    "NAdam_CCD_42_1",
    "NAdam_CCD_43_1",
    "NAdam_CCD_44_1",
    "NAdam_CCD_45_1",
    "NAdam_CCD_46_1",
    "NAdam_CCD_47_1",
    "NAdam_CCD_48_1",
    "NAdam_CCD_49_1",
    "NAdam_CCD_50_1",
    "NAdam_CCD_51_1",
    "NAdam_CCD_52_1",
    "NAdam_CCD_53_1",
    "NAdam_CCD_1_2",
    "NAdam_CCD_2_2",
    "NAdam_CCD_3_2",
    "NAdam_CCD_4_2",
    "NAdam_CCD_5_2",
    "NAdam_CCD_6_2",
    "NAdam_CCD_7_2",
    "NAdam_CCD_8_2",
    "NAdam_CCD_9_2",
    "NAdam_CCD_10_2",
    "NAdam_CCD_11_2",
    "NAdam_CCD_12_2",
    "NAdam_CCD_13_2",
    "NAdam_CCD_14_2",
    "NAdam_CCD_15_2",
    "NAdam_CCD_16_2",
    "NAdam_CCD_17_2",
    "NAdam_CCD_18_2",
    "NAdam_CCD_19_2",
    "NAdam_CCD_20_2",
    "NAdam_CCD_21_2",
    "NAdam_CCD_22_2",
    "NAdam_CCD_23_2",
    "NAdam_CCD_24_2",
    "NAdam_CCD_25_2",
    "NAdam_CCD_26_2",
    "NAdam_CCD_27_2",
    "NAdam_CCD_28_2",
    "NAdam_CCD_29_2",
]

for name in name_list
    # Plot with validation data
    #model, binvars, convars, oracle, nbin, ncon, ntotal = load_cobra_model()
    #nn = get_nn(name)
    #y, ŷ = evaluate_nn_cache(nn, n_samples, cachedir)
    
    #output_file =  open("cobra_" * name * ".out", "a")
    #write.([output_file], string.(y) .* "\n")
    #close(output_file)
    #output_file =  open("eval_" * name * ".out", "a")
    #write.([output_file], string.(ŷ) .* "\n")
    #close(output_file)

    #create_eval_plots(name, y, ŷ)

    # Calculate mean absolute error
    stats = get_stats(name)
    println(get_absolute_error(stats, 10))
end