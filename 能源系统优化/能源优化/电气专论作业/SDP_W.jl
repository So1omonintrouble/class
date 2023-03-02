using JuMP, MosekTools, LinearAlgebra
# 拓扑关系
branch=[#序号 节点1 节点2 支路容量
            1   1   2;
            2   2   3;
            3   3   4;
            4   4   5;
            5   5   6;
            6   6   7;
            7   7   1;
            8   1   8;
            9   2   8;
            10  2   4;
            11  7   8;
            12  2   5;
            13  2   6;
            14  3   5;
            15  3   6;
            16  4   6;
            17  6   8
]
bus = 1:8
n = size(bus,1)
(m,nons) = size(branch)
eye = Matrix(I, n, n)
Column_u_v = ones(n,1)

# 尝试建立节点节点关联A矩阵但失败，不知原因
# 因为标点错了
A = zeros(n,n)
for i=1:n
   for j=1:n
       for h=1:m
           if branch[h,1]==i&&branch[h,2]==j
               A[i,j]==1
           end
        end
    end
end
@constraint(W_SDP_model,[i=1:8,j=1:8,h=1:22;],W)

# 创建模型
SDP_W_model=Model(with_optimizer(Mosek.Optimizer))
# 创建变量
@variable(SDP_W_model, W[1:n,1:n], PSD)
# 最优W
@variable(SDP_W_model, s >= 0)
# s
@variable(SDP_W_model, A[1:n,1:n])
# 优化目标为最小化W的最大特征值的绝对值s
@objective(SDP_W_model, Min, s)
# 约束条件
@constraint(SDP_W_model,W*Column_u_v .== Column_u_v )
# W1 = 1
@constraint(SDP_W_model,
            Symmetric([s.*eye W-ones(n,n)/n;W-ones(n,n)/n s.*eye]) in PSDCone())
# 半定约束
 @constraint( SDP_W_model, [i=1:n,j=1:n; A[i,j] == 0], W[i,j] == 0)
 #                             for ⤒         if ⤒       constraint ⤒
# W ∈ L

# 描述优化模型
print(SDP_W_model)

# 求解优化模型
optimize!(SDP_W_model)

# 输出求解结果
println("程序终止状态：", termination_status(SDP_W_model))

# 目标函数值和变量值
println("目标函数值：", objective_value(SDP_W_model))
println("最优 W 矩阵：", value.(W))
