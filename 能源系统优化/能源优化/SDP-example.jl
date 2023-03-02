using JuMP,MosekTools,LinearAlgebra
C = [1 0;0 4]
A = [5 6;7 8]
b = 9
model=Model(Mosek.Optimizer)
# 方法一：直接建立半定矩阵变量
@variable(model,X[1:2,1:2],PSD)
# 方法二：利用半定锥构建半定矩阵变量
# @variable(model,X[1:2,1:2] in PSDCone())
# 方法三：先定义矩阵变量，再增加半定约束
# @variable(model,X[1:2,1:2],Symmetric)
# @constraint(model,X>=0,PSDCone())
@objective(model,Min,dot(C,X))
@constraint(model,dot(A,X)==b)
print(model)
optimize!(model)
println("程序终止状态：", termination_status(model))
println("原问题状态：", primal_status(model))
println("最优值：",objective_value(model))
println("最优解：X = ",value.(X))
