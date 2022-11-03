using Plots
using DataFrames
using Flux
using BSON

ENV["GKSwstype"] = "100"

function save_test_train(stats, filename)
    p_train = plot(stats.epoch, hcat(stats.train_mean, stats.train_max),
        label=["Average" "Max"], legend=:left, xlabel="Epoch", 
        ylabel="Train Loss (MSE)", yaxis=:log)
    p_test = plot(stats.epoch, hcat(stats.test_mean, stats.test_max), 
        legend=false, xlabel="Epoch", ylabel="Test Loss (MSE)", yaxis=:log)
    plot!(p_train, p_test)
    savefig(filename)
end

function save_orderplot(ŷ, y, filename)
    I = sortperm(y)
    plot(1:length(y), hcat(y[I], ŷ[I]), label=["Actual" "Predicted"], 
        legend=:bottomright, xlabel="Sample", ylabel=("Fitness"))
    savefig(filename)
end

function save_lrplot(stats, filename)
    if "learning_rate" in names(stats)
        lr = stats.learning_rate
        plot(stats.epoch, lr, legend=false, xlabel="Epoch", ylabel="Learning Rate", 
        yaxis=:log)
        savefig(filename)
    end
    
end

function create_run_plots(name)
    rundir = "runs/" * name * "/"
    epochdir = rundir * "epochs/"
    imgdir = rundir * "img/"
    mkpath(imgdir)
    digits = 4

    for bsonfile in readdir(epochdir)
        filename = epochdir * bsonfile
        epoch = string(parse(Int, splitext(splitdir(filename)[2])[1]))
        while length(epoch) < digits
            epoch = "0" * epoch
        end
        bson = BSON.load(filename)
        stats = bson["stats"]
        save_test_train(stats, imgdir * epoch * "_test_train.png")
        ŷ = bson["ŷ"]
        y = bson["y"]
        save_orderplot(ŷ, y, imgdir * epoch * "_orderplot.png")
        save_lrplot(stats, imgdir * epoch * "_lrplot.png")
    end
end