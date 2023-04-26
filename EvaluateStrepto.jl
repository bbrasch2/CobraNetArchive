include("StreptoNet.jl")

name_list = [
    "hypertest_lr_0.001_decay_0.995_start_100_l1_1.0e-5_l2_1.0e-5_dropout_0.1",
    "sigmoid_test",
    "relu_test"
]

for name in name_list
    train_error, test_error = get_absolute_error(get_stats(name), 10)
    println(name)
    println("Train error: " * string(train_error))
    println("Test error: " * string(test_error))
    println()
end