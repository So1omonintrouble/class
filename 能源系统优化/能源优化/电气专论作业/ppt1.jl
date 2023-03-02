using JuMP,GLPK
c = [ 1 1 1 1 1 100 100 1000 1000 1000 ]
Pmax = 2000
Qmax = 1400
Pi = [ 104	130	210	222	310	261	362	135	384	453 ]
Qi = [ 72	93	159	143	279	190	286	86	312	321 ]
SFR = Model(with_optimizer(GLPK.Optimizer))
@variable(SFR,  x[1:10], Bin)
@variable(SFR, P)
@objective(SFR, Max, sum(c'.*x))
@constraint(SFR,0 <= P <= Pmax)
@constraint(SFR,P == sum(x'.*Pi))
optimize!(SFR)
objective_value(SFR)
value.(x)
value.(P)
# print(SFR)
println("程序终止状态：", termination_status(SFR))
println("恢复情况：", value.(x))
