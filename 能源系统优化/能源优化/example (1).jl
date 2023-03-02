using JuMP,MosekTools

model1 = Model(Mosek.Optimizer)
@variable(model1,x[1:2])
@objective(model1,Min,x[1]+x[2])
@constraint(model1,con1,x[1]^2+x[2]^2<=1)
print(model1)
optimize!(model1)
println("最优值：",objective_value(model1))
println("最优解：",value.(x))

model2 = Model(Mosek.Optimizer)
@variable(model2,x[1:2])
@objective(model2,Min,sum(x))
@constraint(model2,con2,[[1];x] in SecondOrderCone())
print(model2)
optimize!(model2)
println("最优值：",objective_value(model2))
println("最优解：",value.(x))
