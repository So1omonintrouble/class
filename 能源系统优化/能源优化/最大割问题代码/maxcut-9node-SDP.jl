using JuMP, MosekTools, LinearAlgebra
W=[
0 10 0 0 0 11 0 0 0
10 0 18 0 0 0 16 0 12
0 18 0 22 0 0 0 0 8
0 0 22 0 20 0 24 16 21
0 0 0 20 0 26 0 7 0
11 0 0 0 26 0 17 0 0
0 16 0 24 0 17 0 19 0
0 0 0 16 7 0 19 0 0
0 12 8 21 0 0 0 0 0
]
w=sum(W)/4
W_SDP_model=Model(Mosek.Optimizer)
@variable(W_SDP_model, Y[1:9,1:9], PSD)
@objective(W_SDP_model, Max, w-dot(W,Y)/4)
@constraint(W_SDP_model,[i=1:9],Y[i,i]==1)
print(W_SDP_model)
optimize!(W_SDP_model)
println("程序终止状态：", termination_status(W_SDP_model))
println("目标函数值：", objective_value(W_SDP_model))
println("Y矩阵值：", value.(Y))
