using JuMP,GLPK,LinearAlgebra
b  = [ 0  4  6]
DtoO = Model(with_optimizer(Mosek.Optimizer))
@variable(DtoO, x[1:3])
@objective(DtoO, Max, sum(b*x))
@constraint(DtoO,con1,2*x[1]+x[2]-x[3]>=1)
@constraint(DtoO,con2,3*x[1]+2*x[2]-x[3]<=-4)
@constraint(DtoO,con3,-x[1]+3*x[2]+2*x[3]<=0)
@constraint(DtoO,con4,x[1]+4*x[2]+x[3]==0)

@constraint(DtoO,con5, x[1].<=0)
@constraint(DtoO,con6, x[2].>=0)
optimize!(DtoO)
objective_value(DtoO)
value.(x)
print(DtoO)
println("程序终止状态：", termination_status(DtoO))
println("原问题状态：", primal_status(DtoO))
# 目标函数值和变量值
println("目标函数值：", objective_value(DtoO))
println("变量x值：", value.(x))

a=[1 0 2;
   0 1 3]
(m,n) = size(a)
b=zeros(3m,3n)
   for i=1:m
      for im=1:n
         for h=1:3
            b[3i+h-3,3im+h-3]=a[i,im]
         end
      end
   end
println(a)
