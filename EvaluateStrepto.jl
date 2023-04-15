include("StreptoNet.jl")

name_list = [
    "decay_0.99"
]

for name in name_list
    train_error, test_error = get_absolute_error(get_stats(name), 10)
    println(name)
    println("Train error: " * string(train_error))
    println("Test error: " * string(test_error))
    println()
end