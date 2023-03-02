using JuMP,MosekTools

bus=[1 100 60 800 600
     2 90 40 0 0
     3 120 80 570 500
     4 60 30 0 0
     5 60 20 0 0
     6 200 100 510 440
     7 200 100 0 0
     8 60 20 0 0
     9 60 20 0 0
     10 45 30 0 0
     11 60 35 0 0
     12 60 35 0 0
     13 120 80 0 0
     14 60 10 420 380
     15 60 20 0 0
     16 60 20 0 0
     17 90 40 0 0
     18 90 40 0 0
     19 90 40 0 0
     20 90 40 380 340
     21 90 40 0 0
     22 90 40 0 0
     23 420 200 0 0
     24 420 200 600 500
     25 60 25 0 0
     26 60 25 0 0
     27 60 20 400 320
     28 120 10 0 0
     29 200 600 0 0
     30 150 70 0 0
     31 210 100 490 450
     32 60 40 0 0]

branch=[1 2 0.493 0.2511
        2 3 0.366 0.1864
        3 4 0.3811 0.1941
        4 5 0.819 0.707
        5 6 0.1872 0.6188
        6 7 0.7114 0.2351
        7 8 1.03 0.74
        8 9 1.044 0.74
        9 10 0.1966 0.065
        10 11 0.3744 0.1238
        11 12 1.468 1.155
        12 13 0.5416 0.7129
        13 14 5.91 5.26
        14 15 7.463 5.45
        15 16 3.289 4.721
        16 17 7.32 5.74
        1 18 0.164 0.1565
        18 19 1.5042 1.3554
        19 20 0.4095 0.4784
        20 21 0.7089 0.9373
        2 22 4.512 3.083
        22 23 0.898 0.7091
        23 24 0.896 0.7091
        5 25 0.203 0.1034
        25 26 0.2842 0.1447
        26 27 1.059 0.9337
        27 28 0.8042 0.7006
        28 29 0.5075 0.2585
        29 30 0.9744 0.963
        30 31 0.3105 0.3619
        31 32 0.341 0.5302]

n=size(bus,1)#节点数
m=size(branch,1)#支路数
P_load=bus[:,2]*1000#有功负荷
Q_load=bus[:,3]*1000#无功负荷
Pmax=bus[:,4]*1000#发电机有功上限
Qmax=bus[:,5]*1000#发电机无功上限
r=branch[:,3]#支路电阻
x=branch[:,4]#支路电抗
V0=12660#节点电压
range=0.015#节点电压波动范围
lmax=100#电流上限
model=Model(with_optimizer(Mosek.Optimizer))
@variable(model,P[1:m])
@variable(model,Q[1:m])
@variable(model,p[1:n])
@variable(model,q[1:n])
@variable(model,v[1:n])
@variable(model,l[1:m])

@objective(model,Min,sum(p))#最小化有功网损
#潮流方程
@constraint(model,[j in 1:n],p[j]-P_load[j]==sum(P[jk] for jk in findall(branch[:,1].==j)) - sum(P[ij]-r[ij]*l[ij]  for ij in findall(branch[:,2].==j) ))
@constraint(model,[j in 1:n],q[j]-Q_load[j]==sum(Q[jk] for jk in findall(branch[:,1].==j)) - sum(Q[ij]-x[ij]*l[ij]  for ij in findall(branch[:,2].==j) ))
#欧姆定律
for ij = 1:m
    i=Int(branch[ij,1])
    j=Int(branch[ij,2])
    @constraint(model,v[j]==v[i]-2*(r[ij]*P[ij]+x[ij]*Q[ij])+(r[ij]^2+x[ij]^2)*l[ij])
end
#二阶锥约束
for ij = 1:m
    i=Int(branch[ij,1])
    @constraint(model,[l[ij]+v[i],2*P[ij],2*Q[ij],l[ij]-v[i]] in SecondOrderCone())
end
#节点电压约束
@constraint(model,[i in 1:n],(V0*(1-range))^2<=v[i]<=(V0*(1+range))^2)
#发电机出力限制
@constraint(model,[i in 1:n],0<=p[i]<=Pmax[i])
@constraint(model,[i in 1:n],0<=q[i]<=Qmax[i])
#电流约束
@constraint(model,[i in 1:m],l[i]<=lmax^2)

optimize!(model)
println(termination_status(model))

println(objective_value(model)-sum(P_load))
println("DG1出力:",value(p[24])/1000,"kW")
println("DG2出力:",value(p[20])/1000,"kW")
println("DG3出力:",value(p[3])/1000,"kW")
println("DG4出力:",value(p[6])/1000,"kW")
println("DG5出力:",value(p[27])/1000,"kW")
println("DG6出力:",value(p[31])/1000,"kW")
println("DG7出力:",value(p[1])/1000,"kW")
println("DG8出力:",value(p[14])/1000,"kW")
println("DG1出力:",value(q[24])/1000,"kvar")
println("DG2出力:",value(q[20])/1000,"kvar")
println("DG3出力:",value(q[3])/1000,"kvar")
println("DG4出力:",value(q[6])/1000,"kvar")
println("DG5出力:",value(q[27])/1000,"kvar")
println("DG6出力:",value(q[31])/1000,"kvar")
println("DG7出力:",value(q[1])/1000,"kvar")
println("DG8出力:",value(q[14])/1000,"kvar")
