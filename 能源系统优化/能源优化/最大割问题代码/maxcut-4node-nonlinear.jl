using JuMP,Ipopt,Juniper,LinearAlgebra
W = [
    0 4 12 0
    4 0 9 14
    12 9 0 7
    0 14 7 0
    ]
optimizer = Juniper.Optimizer
nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)
model = Model(optimizer_with_attributes(optimizer, "nl_solver"=>nl_solver))
@variable(model, x[1:4], Bin)
@objective(model, Max, 1/4*dot(W,ones(4,4)-(2x.-1)*(2x.-1)'))
print(model)
optimize!(model)
println("程序终止状态：", termination_status(model))
println("目标函数值：", objective_value(model))
println("优化变量：", value.(x))
