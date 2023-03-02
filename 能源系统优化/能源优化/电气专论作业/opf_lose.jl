using JuMP, MosekTools,LinearAlgebra
branch=[#序号 节点1 节点2     电阻        电抗
        1    1      2       0.493       0.2511;
        2    2      3       0.366       0.1864;
        3    3      4       0.3811      0.1941;
        4    4      5       0.819       0.707;
        5    5      6       0.1872      0.6618;
        6    6      7       0.7114      0.2351;
        7    7      8       1.03        0.74;
        8    8      9       1.044       0.74;
        9    9      10      0.1966      0.065;
        10   10     11      0.3744      0.1238;
        11   11     12      1.468       1.155;
        12   12     13      0.5416      0.7129;
        13   13     14      5.19        5.26;
        14   14     15      7.463       6.45;
        15   15     16      3.289       4.721;
        16   16     17      7.32        5.74;
        17   1      18      0.164       0.1565;
        18   18     19      1.5042      1.3554;
        19   19     20      0.4095      0.4784;
        20   20     21      0.7089      0.9373;
        21   2      22      4.512       3.083;
        22   22     23      0.898       0.7091;
        23   23     24      0.896       0.7011;
        24   5      25      0.203       0.1034;
        25   25     26      0.2842      0.1447;
        26   26     27      1.059       0.9337;
        27   27     28      0.8042      0.7006;
        28   28     29      0.5075      0.2585;
        29   29     30      0.9744      0.963;
        30   30     31      0.3105      0.3619;
        31   31     32      0.341       0.5302
]
bus=[#序号 有功负荷 无功负荷 有功上限 无功上限
    1       100     60      800     600;
    2       90      40      0       0;
    3       120     80      570     500;
    4       60      30      0       0;
    5       60      20      0       0;
    6       200     100     510     440;
    7       200     100     0       0;
    8       60      20      0       0;
    9       60      20      0       0;
    10      45      30      0       0;
    11      60      35      0       0;
    12      60      35      0       0;
    13      120     80      0       0;
    14      60      10      420     380;
    15      60      20      0       0;
    16      60      20      0       0;
    17      90      40      0       0;
    18      90      40      0       0;
    19      90      40      0       0;
    20      90      40      380     340;
    21      90      40      0       0;
    22      90      50      0       0;
    23      420     200     0       0;
    24      420     200     600     500;
    25      60      25      0       0;
    26      60      25      0       0;
    27      60      20      400     320;
    28      120     10      0       0;
    29      200     600     0       0;
    30      150     70      0       0;
    31      210     100     490     450;
    32      60      40      0       0
    ]
  #节点数n以及支路数m
(n,nons)=size(bus)
(m,nons)=size(branch)
  # 根据假定方向创建节点支路关联矩阵A
A1=zeros(n,m)#父节点矩阵
A2=zeros(n,m)#子节点矩阵
for i=1:n
    for j=1:m
        if branch[j,2]==i
            A1[i,j]=1  # 每个节点都是相关支路的父节点
        end
        if branch[j,3]==i
            A2[i,j]=1  # 每个节点都是相关支路的子节点
        end
    end
end
 B1 = A1'
 B2 = A2'
#线路参数
r=branch[:,4]
x=branch[:,5]
#节点负载参数
p_d=bus[:,2].*100
q_d=bus[:,3].*100
#节点发电机参数
p_iM=bus[:,4].*100
q_iM=bus[:,5].*100
#电压波动范围
range = 0.015
#额定电压
V_n = 12660
###############################建模##########################
DN_model=Model(with_optimizer(Mosek.Optimizer))
###############################变量#############################
# 节点电压的平方
@variable(DN_model,V[1:n])
# 支路电流的平方
@variable(DN_model,I[1:m])
# 节点注入功率
@variable(DN_model,p_i[1:n])
@variable(DN_model,q_i[1:n])
#节点流出线路的功率
@variable(DN_model,P[1:m])
@variable(DN_model,Q[1:m])
# 优化目标为最小化网损，一般用电阻和线路电流表示
#############################优化目标#############################
@objective(DN_model,Min,r'*I)
#############################约束条件#############################
# 节点电压约束
@constraint(DN_model, (V_n*(1-range))^(2).<= V .<= (V_n*(1+range))^(2) )
# 支路电流约束
@constraint(DN_model, 0 .<= I .<= 10000 )
# 节点功率约束：发电机流入该节点的功率-该节点负载消耗+上条支路流进功率-流向下条支路功率
@constraint(DN_model, p_i - p_d + A2*(P+r.*I) - A1*P .== 0 )
@constraint(DN_model, q_i - q_d + A2*(Q+x.*I) - A1*Q .== 0 )
# 线路欧姆定律：懒得解释了orz
@constraint(DN_model, B1*V - B2*V .== 2*(r.*P+x.*Q) - (r.*r+x.*x).*I)
#变电站功率
@constraint(DN_model, 0 .<= p_i .<= p_iM )
@constraint(DN_model, 0 .<= q_i .<= q_iM )
#二阶锥约束
@constraint(DN_model,[i=1:m],[V[Int(branch[i,2])]+I[i],2*P[i],2*Q[i],V[Int(branch[i,2])]-I[i]] in SecondOrderCone())
###############################解决问题##############################
# 求解优化模型
optimize!(DN_model)
# 打印优化模型
print(DN_model)
# 求解状态
println( termination_status(DN_model))
# 目标值
objective_value(DN_model)#
println("最优发电策略为各个DG的有功出力分别是：",value.(p_i))
println("最优发电策略为各个DG的无功出力分别是：",value.(q_i))
println("最小网损：",objective_value(DN_model))
