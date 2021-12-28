
using JLD
using Plots
using DataFrames

run_name = "testing"
rundir = "/home/paul/CobraNet/runs/" * run_name * "/"

function save_test_train(stats, filename)
    p_train = plot(stats.epoch, hcat(stats.train_mean, stats.train_max), plot_title="train", ylim=(0,1), legend=false)
    p_test = plot(stats.epoch, hcat(stats.test_mean, stats.test_max), plot_title="test", ylim=(0,1), legend=false)
    plot!(p_train, p_test, plot_title="")
    savefig(filename)
end

function save_orderplot(ŷ, y, filename)
    I = sortperm(y)
    plot(1:length(y), hcat(y[I], ŷ[I]), label=["y" "ŷ"], 
         legend=:bottomright)
    savefig(filename)
end

epochdir = rundir * "epochs/"
imgdir = rundir * "img/"
mkpath(imgdir)

for jldfile in readdir(epochdir)
    filename = epochdir * jldfile
    epoch = parse(Int, splitext(splitdir(filename)[2])[1])
    stats = load(filename, "stats")
    save_test_train(stats, imgdir * "test_train_" * string(epoch) * ".png")
    ŷ = load(filename, "ŷ")
    y = load(filename, "y") 
    save_orderplot(ŷ, y, imgdir * "orderplot_" * string(epoch) * ".png")
end

