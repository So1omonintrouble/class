using JuMP, MosekTools
# 创建模型
lmp_model = Model(Mosek.Optimizer)

# 变量g1和g2分别表示发电机G1和G2的出力，l1表示线路功率
@variable(lmp_model,g1>=0)
@variable(lmp_model,g2>=0)
@variable(lmp_model,l1)

# 优化目标为最小化总购电成本
@objective(lmp_model,Min,300g1+500g2)

# 约束条件
# 节点功率平衡约束
@constraint(lmp_model,pb1,g1-l1==50)
@constraint(lmp_model,pb2,g2+l1==200)
# 线路容量约束
@constraint(lmp_model,lc_max,l1<=100)
@constraint(lmp_model,lc_min,l1>=-100)
# 发电机容量约束
@constraint(lmp_model,gc1,g1<=300)
@constraint(lmp_model,gc2,g2<=300)

# 打印优化模型
print(lmp_model)

# 求解优化模型
optimize!(lmp_model)

# 输出求解结果
# 求解状态
println("程序终止状态：", termination_status(lmp_model))
println("原问题状态：", primal_status(lmp_model))
println("对偶问题状态：", dual_status(lmp_model))
# 目标函数值和变量值
println("目标函数值：", objective_value(lmp_model))
println("发电机G1出力：", value.(g1))
println("发电机G2出力：", value.(g2))
println("线路传输功率：", value.(l1))

# 对偶变量
println("是否存在对偶变量：", has_duals(lmp_model))
println("节点A功率平衡约束的对偶变量值：", dual(pb1))
println("节点B功率平衡约束的对偶变量值：", dual(pb2))
println("线路传输功率上界约束的对偶变量值：", dual(lc_max))
println("线路传输功率下界约束的对偶变量值：", dual(lc_min))
println("发电机G1容量约束的对偶变量值：", dual(gc1))
println("发电机G2容量约束的对偶变量值：", dual(gc2))

# 影子价格
println("节点A功率平衡约束的影子价格：", shadow_price(pb1))
println("节点B功率平衡约束的影子价格：", shadow_price(pb2))
println("线路传输功率上界约束的影子价格：", shadow_price(lc_max))
println("线路传输功率下界约束的影子价格：", shadow_price(lc_min))
println("发电机G1容量约束的影子价格：", shadow_price(gc1))
println("发电机G2容量约束的影子价格：", shadow_price(gc2))
