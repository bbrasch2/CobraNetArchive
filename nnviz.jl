
using JLD
using Plots
using DataFrames

function plot_test_train(stats; epoch=0)
    if epoch < 1
        epoch = nrow(stats)
    end
    p_train = plot(1:epoch, hcat(stats.train_mean[1:epoch], stats.train_max[1:epoch]), plot_title="train", ylim=(0,1), legend=false)
    p_test = plot(1:epoch, hcat(stats.test_mean[1:epoch], stats.test_max[1:epoch]), plot_title="test", ylim=(0,1), legend=false)
    display(plot!(p_train, p_test, plot_title=""))
end

epoch = 280
stats = load("saved" * string(epoch) * ".jld", "stats")
plot_test_train(stats; epoch)
