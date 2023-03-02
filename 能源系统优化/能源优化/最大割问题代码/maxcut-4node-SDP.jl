using JuMP,MosekTools,LinearAlgebra
W = [
    0 4 12 0
    4 0 9 14
    12 9 0 7
    0 14 7 0
    ]
w = sum(W)/4
W_SDP_model=Model(Mosek.Optimizer)
@variable(W_SDP_model, Y[1:4,1:4], PSD)
@objective(W_SDP_model, Max, w-dot(W,Y)/4)
@constraint(W_SDP_model,[i=1:4],Y[i,i]==1)
print(W_SDP_model)
optimize!(W_SDP_model)
println("程序终止状态：", termination_status(W_SDP_model))
println("目标函数值：", objective_value(W_SDP_model))
println("Y矩阵值：", value.(Y))
