using Plots
using DataFrames
using Flux
using BSON
using Statistics
using LinearAlgebra

ENV["GKSwstype"] = "100"

function save_test_train(stats, filename)
    p_train = plot(stats.epoch, hcat(stats.train_mean, stats.train_max),
        label=["Average" "Max"], legend=:left, xlabel="Epoch", 
        ylabel="Train Error")#, ylims=(0, 1)) #, yaxis=:log)
    p_test = plot(stats.epoch, hcat(stats.test_mean, stats.test_max), 
        legend=false, xlabel="Epoch", ylabel="Test Error")#, ylims=(0, 1)) #, yaxis=:log)
    plot!(p_train, p_test)
    savefig(filename)
end

function save_orderplot(ŷ, y, filename, epoch)
    I = sortperm(y)
    plot_title = "Epoch: " * epoch
    plot(1:length(y), hcat(y[I], ŷ[I]), label=["Actual" "Predicted"], 
        legend=:bottomright, xlabel="Sample", ylabel=("Fitness"),# ylims=(-0.1, 1.1),
        title=plot_title)
    savefig(filename)
end

function save_lrplot(stats, filename)
    if "learning_rate" in names(stats)
        lr = stats.learning_rate
        lr[lr.<=0] .= NaN
        plot(stats.epoch, lr, legend=false, xlabel="Epoch", ylabel="Learning Rate", 
        yaxis=:log)
        savefig(filename)
    end
    
end

function create_run_plots(name; plot_lr=true)
    rundir = "streptoruns/" * name * "/"
    epochdir = rundir * "epochs/"
    imgdir = rundir * "img/"
    mkpath(imgdir)
    digits = 4

    files = readdir(epochdir)
    temp_files = []
    for file in files
        while length(file) < digits + length(".bson")
            file = "0" * file
        end
        push!(temp_files, file)
    end
    order = sortperm(temp_files)
    files = files[order]

    anim = @animate for bsonfile in files
        filename = epochdir * bsonfile
        epoch = string(parse(Int, splitext(splitdir(filename)[2])[1]))
        while length(epoch) < digits
            epoch = "0" * epoch
        end
        bson = BSON.load(filename)
        stats = bson["stats"]
        ŷ = bson["ŷ"]
        y = bson["y"]
        save_test_train(stats, imgdir * epoch * "_test_train.png")
        if plot_lr
            save_lrplot(stats, imgdir * epoch * "_lrplot.png")
        end
        save_orderplot(ŷ, y, imgdir * epoch * "_orderplot.png", string(epoch))
    end

    gif(anim, imgdir * "orderplots.gif", fps = 3)
end