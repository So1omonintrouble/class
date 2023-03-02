# Phasor Estimate for PMU, 2019-3-15, By Prof. Yin Xu
#    for the "Smart Grid" course @ BJTU
# Homework for Week 3
# Phasor Estimate for PMU, 2019-3-30,Added By Stu. Yanran Du

# Sampling
f0 = 50
fs = 3200
deltaT = 1/fs
# 0到40ms一共64*2+1个点，我们要先有64个点才能开始计算，第64个点正好是在20-1/3.2ms处，一共可以进行66次计算
#(💠很多同学在时间段这里都有问题💠)
t = 0:deltaT:0.04
# 根据同步向量的概念：同步相量 𝑿(𝑡) 的 相角 是𝑥(t)相对于与UTC同步且频率为额定频率的余弦函数 cos𝜔0𝑡 的瞬时相角(PPT—15)；幅值/根号2👉有效值(PPT—14)。可以注意到，同步向量可表达为的准确的相角应为π/4，即45°；有效值应为7.07…，即10/根号2。
# 输入\alpha可以得到α【\+对应英文名称】
x = 10cos.(2pi*f0*t.+pi/4)

# Estimate phasors with non-recursive DFT
N = Int(fs/f0)
θ = 2pi/N
X_DFT_nr = complex(zeros(size(x,1)-N+1))# 含有66个0元素的复数向量
X_DFT_polar_nr = zeros(size(x,1)-N+1,2)# 含有66个0元素的复数向量
for i in 1:size(X_DFT_nr,1)
    X_DFT_nr[i] = sqrt(2)/N*sum(x[i+n]*exp(-im*n*θ) for n = 0:N-1)
    X_DFT_polar_nr[i,:] = [abs(X_DFT_nr[i]),angle(X_DFT_nr[i])*180/pi]
end
println([X_DFT_polar_nr])
#调出相角每次测量曲线图，可以自己尝试调出相应幅值曲线图
using Plots,Plotly
m = 1:66
plt1 = Plotly.plot(m,X_DFT_polar_nr[m,2])
display(plt1)
println(" ")
println("第一个角度为0初始时刻的𝛗,故非递归DFT法下估计同步向量X(t)为", [X_DFT_polar_nr[1, :]])

# Estmate synchrophasors with recursive DFT
X_DFT_r = complex(zeros(size(x,1)-N+1))
X_DFT_polar_r = zeros(size(x,1)-N+1,2)
# Take the initial value from the previous result
X_DFT_r[1] = X_DFT_nr[1]
X_DFT_polar_r[1,:] = X_DFT_polar_nr[1,:]
for i in 2:size(X_DFT_r,1)
    X_DFT_r[i] = X_DFT_r[i-1] + sqrt(2)/N*(x[i+N-1]-x[i-1])*exp(-im*(i-2)*θ)
    X_DFT_polar_r[i,:] = [abs(X_DFT_r[i]),angle(X_DFT_r[i])*180/pi]
end
# 只需using一次相应pkg，可以统一在程序最开始用逗号隔开各个pkg调用相应pkg
#调出相角每次测量曲线图，可以自己尝试调出相应幅值曲线图
m = 1:66
plt2 = Plotly.plot(m,X_DFT_polar_r[m,2])
display(plt2)
println(" ")
println([X_DFT_polar_r])
println(" ")
println("递归DFT法下估计同步向量X(t)为固定值", [X_DFT_polar_r[2,:]])

# Estmate synchrophasors with recursive DFT(f_1=f0+dev_1, dev_1=0.1Hz)
dev_1 = 0.1
f_1 = f0 + dev_1
x_1 = 10cos.(2pi*f_1*t.+pi/4)
X_DFT_r_1 = complex(zeros(size(x,1)-N+1))
X_DFT_polar_r_1 = zeros(size(x,1)-N+1,2)
# Take the initial value from the previous result
X_DFT_r_1[1] = X_DFT_nr[1]
X_DFT_polar_r_1[1,:] = X_DFT_polar_nr[1,:]
for i in 2:size(X_DFT_r_1,1)
    X_DFT_r_1[i] = X_DFT_r_1[i-1] + sqrt(2)/N*(x_1[i+N-1]-x_1[i-1])*exp(-im*(i-2)*θ)
    X_DFT_polar_r_1[i,:] = [abs(X_DFT_r_1[i]),angle(X_DFT_r_1[i])*180/pi]
end
# 调出相角每次测量曲线图
plt3_p = Plotly.plot(m,X_DFT_polar_r_1[m,2])
display(plt3_p)
#  调出幅值每次测量曲线图
plt3_m = Plotly.plot(m,X_DFT_polar_r_1[m,1])
display(plt3_m)
println(" ")
println([X_DFT_polar_r_1])
println(" ")
println("误差为0.1Hz时，递归DFT法下估计同步向量X(t)为", [X_DFT_polar_r_1[2,:]])
println("根据相角变化曲线图可以发现符合PPT-16的规律，根据PPT-15，相角逐渐增大但是基本接近45°；根据图示幅值变化也较小")

dev_2 = 5
f_2 = f0 + dev_2
x_2 = 10cos.(2pi*f_2*t.+pi/4)
X_DFT_r_2 = complex(zeros(size(x,1)-N+1))
X_DFT_polar_r_2 = zeros(size(x,1)-N+1,2)
# Take the initial value from the previous result
X_DFT_r_2[1] = X_DFT_nr[1]
X_DFT_polar_r_2[1,:] = X_DFT_polar_nr[1,:]
for i in 2:size(X_DFT_r_2,1)
    X_DFT_r_2[i] = X_DFT_r_2[i-1] + sqrt(2)/N*(x_2[i+N-1]-x_2[i-1])*exp(-im*(i-2)*θ)
    X_DFT_polar_r_2[i,:] = [abs(X_DFT_r_2[i]),angle(X_DFT_r_2[i])*180/pi]
end
# 调出相角每次测量曲线图
plt4_p = Plotly.plot(m,X_DFT_polar_r_2[m,2])
display(plt4_p)
# 调出相应幅值曲线图
plt4_m = Plotly.plot(m,X_DFT_polar_r_2[m,1])
display(plt4_m)
println(" ")
println([X_DFT_polar_r_2])
println(" ")
println("误差为5Hz时，递归DFT法下估计同步向量X(t)为", [X_DFT_polar_r_2[2,:]])
println("可以发现符合PPT-16的规律，根据PPT-15，相角误差逐渐增大且远远超过45°")
println("同样根据图线可以发现幅值的测量也出现很大偏差")
