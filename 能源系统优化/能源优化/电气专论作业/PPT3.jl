using JuMP,GLPK
c = [ 3 1 5 3 2 1 1 ]
fm = [ 300 300 300 300 500 600 350 ]
P = [ 400 -300 300 450 -500 0 -350 ]
branch = [3     2   1
          1     2   2
          1     6   3
          2     6   4
          4     5   5
          6     5   6
          5     7   7]
A = zeros(7,7)
    for i = 1:7
        for j = 1:7
            if branch[j,1] == i
                A[i,j] = -1   # 流出为负
            end
            if branch[j,2] == i
                A[i,j] = 1  # 流入为正
            end
        end
    end
NP = Model(with_optimizer(GLPK.Optimizer))
@variable(NP, f[1:7])
@objective(NP, Max, sum(c'.*f))
@constraint(NP, 0 .<= f .<= fm')
@constraint(NP, P' + A*f .== 0)
optimize!(NP)
objective_value(NP)
value.(f)
print(NP)
println("程序终止状态：", termination_status(NP))
println("恢复情况：", value.(f))
