using Plots,JuMP,GLPK
plotly()

E = [
    20.
    20
    19
    18
    17
    17
    19
    20
    21
    22
    24
    25
    27
    28
    28
    28
    27
    26
    25
    23
    22
    21
    20
    19
]

# E .+= 5

p = plot(reshape(E,24,1), lw=3, size=(1200,900),
    xlabel="Hour Ending", ylabel="Outdoor Tempurature (°C)",
    leg = false,
    xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,
    yguidefontsize=16,titlefontsize=16)
scatter!(p,reshape(E,24,1),leg = false)
savefig(p,"outdoor_tempurate.png")

αᴴ=0.94
αᴾ=0.5
Pmax=1
Tmax=22
Tmin=20
T0=21

c = ones(24,1)*0.7847
c[11:15] .= 1.3104
c[19:21] .= 1.3104
c[1:7] .= c[24] = 0.3113

dr = Model(GLPK.Optimizer)
@variable(dr,P[1:24])
@variable(dr,T[1:24])
@variable(dr,s[1:24])
@objective(dr,Min,sum(c.*s))
@constraint(dr,[t in 1:24],s[t] >= P[t])
@constraint(dr,[t in 1:24],s[t] >= -P[t])
@constraint(dr,T[1]==αᴴ*T0+(1-αᴴ)*E[1]+αᴾ*P[1])
@constraint(dr,[t in 2:24],T[t]==αᴴ*T[t-1]+(1-αᴴ)*E[t]+αᴾ*P[t])
@constraint(dr,[t in 1:24],-Pmax<=P[t]<=Pmax)
@constraint(dr,[t in 1:24],Tmin<=T[t]<=Tmax)

optimize!(dr)

# 输出结果
if termination_status(dr) == MOI.OPTIMAL
    P_opt = value.(P)
    T_opt = value.(T)
    println("最优购电成本为：",objective_value(dr))
else
    error("The model was not solved correctly.")
end

# 结果作图
p = plot(reshape(P_opt,24,1), lw=3, size=(800,600),
    xlabel="Hour Ending", ylabel="Load Demand (kW)", label = false,
    xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,
    yguidefontsize=16,titlefontsize=16,legendfontsize=16)
scatter!(p,reshape(P_opt,24,1),label = false)
savefig(p,"load_demand.png")

p = plot(reshape(T_opt,24,1), lw=3, size=(800,600),
    xlabel="Hour Ending", ylabel="Indoor Tempurature (°C)", label = false,
    xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,
    yguidefontsize=16,titlefontsize=16,legendfontsize=16)
scatter!(p,reshape(T_opt,24,1),label = false)
savefig(p,"indoor_tempurature.png")
