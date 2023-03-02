using JuMP,MosekTools,LinearAlgebra
c = [ 1 -4  0  0 ]
OtoD = Model(with_optimizer(Mosek.Optimizer))
@variable(OtoD, x[1:4])
@objective(OtoD, Min, sum(c'.*x))
@constraint(OtoD,con1,2*x[1]+3*x[2]-x[3]+x[4]<=0)
@constraint(OtoD,con2,x[1]+2*x[2]+3*x[3]+4*x[4]>=4)
@constraint(OtoD,con3,-x[1]-x[2]+2*x[3]+x[4]==6)
@constraint(OtoD,con4, x[1]<=0)
@constraint(OtoD,con5, x[2]>=0)
@constraint(OtoD,con6, x[3]>=0)
optimize!(OtoD)
objective_value(OtoD)
value.(x)
print(OtoD)
println("程序终止状态：", termination_status(OtoD))
println("原问题状态：", primal_status(OtoD))
println("对偶问题状态：", dual_status(OtoD))
# 目标函数值和变量值
println("目标函数值：", objective_value(OtoD))
println("变量x值：", value.(x))
# 对偶变量
println("是否存在对偶变量：", has_duals(OtoD))
println("对偶变量值1:", dual(con1))
println("对偶变量值2:", dual(con2))
println("对偶变量值3:", dual(con3))
