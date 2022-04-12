using JLD
using Plots
using DataFrames
using Flux
using BSON

ENV["GKSwstype"] = "100"

function save_test_train(stats, filename)
    p_train = plot(stats.epoch, hcat(stats.train_mean, stats.train_max), plot_title="train", legend=false)
    p_test = plot(stats.epoch, hcat(stats.test_mean, stats.test_max), plot_title="test", legend=false)
    plot!(p_train, p_test, plot_title="")
    savefig(filename)
end

function save_orderplot(ŷ, y, filename)
    I = sortperm(y)
    plot(1:length(y), hcat(y[I], ŷ[I]), label=["y" "ŷ"], 
         legend=:bottomright)
    savefig(filename)
end

function create_run_plots(hyper)
    rundir = "runs/" * hyper.rundir * "/"
    epochdir = rundir * "epochs/"
    imgdir = rundir * "img/"
    mkpath(imgdir)

    for bsonfile in readdir(epochdir)
        filename = epochdir * bsonfile
        epoch = parse(Int, splitext(splitdir(filename)[2])[1])
        bson = BSON.load(filename)
        stats = bson["stats"]
        save_test_train(stats, imgdir * "test_train_" * string(epoch) * ".png")
        ŷ = bson["ŷ"]
        y = bson["y"]
        save_orderplot(ŷ, y, imgdir * "orderplot_" * string(epoch) * ".png")
    end
end