using JuMP,GLPK
c = [ 1 100 1 100 1 100 1000 1000 0 0 0 0 0]
Pgmax = [ 0 0 0 300 500 0 0  0 0 400 0 0 0]
Pi = [100 170 220 128 170 400 229 542 0 0 0 0 0]
S = [1500 700 550 700 700 300 500 300 600 300 700]
branch = [2   3
          3   9
          3   10
          3   12
          4   7
          5   11
          7   2
          9   1
          9   6
          11  2
          12  8]
A = zeros(13,11)
    for i = 1:13
        for j = 1:11
            if branch[j,1] == i
                A[i,j] = -1   # 流出为负
            end
            if branch[j,2] == i
                A[i,j] = 1  # 流入为正
            end
        end
    end
cosϕ = 0.85
sinϕ = sqrt(1-cosϕ^2)
NSFR = Model(with_optimizer(GLPK.Optimizer))
@variable(NSFR, x[1:13], Bin)
@variable(NSFR, Pg[1:13])
@variable(NSFR, p[1:11])
@objective(NSFR, Max, sum(c'.*x))
@constraint(NSFR, 0 .<= Pg .<= Pgmax')
@constraint(NSFR, sum(Pg) == sum(x'.*Pi))
@constraint(NSFR, Pg + A*p .== x.*Pi')
@constraint(NSFR, -S'*cosϕ .<= p .<= S'*cosϕ)
optimize!(NSFR)
objective_value(NSFR)
value.(x)
value.(sum(Pg))
value.(Pg)
value.(p)
# a = value.(p.^2 .+ q.^2)
# b = sqrt.(a)
# print(NSFR)
# println("程序终止状态：", termination_status(NSFR))
println("恢复情况：", value.(x))
println("线路潮流：", value.(p))
println("线路上限：", S*cosϕ)
