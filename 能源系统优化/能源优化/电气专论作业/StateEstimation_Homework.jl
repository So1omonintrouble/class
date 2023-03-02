# State Estimate for PMU, 2019-3-30, By Stu. Yanran Du
#    for the "Smart Grid" course @ BJTU
# Homework for Week 4

using ComplexValues,LinearAlgebra
# 👆可以转换复数为极坐标形式，但反向还不太会，testset里好像有看不太懂😥
#U2 = 1.02*exp(2/180*pi*1im)
#U3 = 1.03*exp(3/180*pi*1im)
# 👆白正同学的表示也很好
polar(r::Real,a::Real) = r*cos(a/180*pi)+r*sin(a/180*pi)*im
# 👆我是用function写的
U_1 = polar(1,0)
U_2 = polar(1.02,2)
U_3 = polar(1.03,3)
x_12 = 0.01im
x_13 = 0.02im
x_23 = 0.015im
println("Answer (1):")
I_12 = (U_1-U_2)/x_12
I_13 = (U_1-U_3)/x_13
I_23 = (U_2-U_3)/x_23
# 只是用一下Polar函数，还是实部虚部更直观
println("I_12 = ",Polar(I_12))
println("I_13 = ",Polar(I_13))
println("I_23 = ",Polar(I_23))
println("I_12 = ",I_12)
println("I_13 = ",I_13)
println("I_23 = ",I_23)
println(" ")

println("Answer (2):")
n = 5*2-1
m = 3*2-1
println("Test_Vector = [U_1r,I_12r,I_12i,U_3r,U_3i,I_13r,I_13i,I_23r,I_23i]")
println("State_Vector = [U_1r,U_2r,U_2i,U_3r,U_3i]")
println("Deviation_Vector = [ε_1,ε_2,ε_3,ε_4,ε_5,ε_6,ε_7,ε_8,ε_9]")
H = zeros(n,m)
# 不知道为啥不往里填1呢❔❓
# 为和大部分同学的H一致更改了State_Vector的元素顺序，因此以下循环不对应目前H
#for i=1:3
#   for j=1:3
#        if i==j
#            H[i,j]==1
#        end
#    end
#end
H[1,1]=1;H[2,4]=1;H[3,5]=1
H[4,3]=-100;H[5,1]=-100;H[5,2]=100
H[6,5]=-50;H[7,1]=-50;H[7,4]=50
H[8,3]=1/0.015;H[8,5]=-1/0.015;H[9,2]=-1/0.015;H[9,4]=1/0.015
println("H = ",H)
println(" ")

println("Answer (3):")
z = [0.995,1.029,0.0545,-2.0,1.9,-2.71,1.0,-1.22,0.598]
X = H'*H\H'*z
println("StateEstimation_Vector=[U_1r,U_2r,U_2i,U_3r,U_3i] = ",X)
Ie_12 = (X[1]-X[2]-X[3]*im)/x_12
Ie_13 = (X[1]-X[4]-X[5]*im)/x_13
Ie_23 = (X[4]+X[5]*im-X[2]-X[3]*im)/x_23
EV = [real(Ie_12),imag(Ie_12),real(Ie_13),imag(Ie_13),real(Ie_23),imag(Ie_23)]
println("[I_12r,I_12i,I_13r,I_13i,I_23r,I_23i] = ",EV)
println(" ")

println("Answer (4):")
real_value=[real(U_1),real(U_3),imag(U_3),real(I_12),imag(I_12),real(I_13),imag(I_13),real(I_23),imag(I_23)]
error = zeros(n,1)
for i=1:n
    error[i]==real_value[i]-z[i]
end
# 👆还是不往里放❓
w = Diagonal([1;1;1;0.1;1;1;0.1;1;1])
println("根据测量值与真实值比较后，令 w = ",w)
X_update = (H'*w*H)\H'*w*z
Ieu_12 = (X_update[1]-X_update[2]-X_update[3]*im)/x_12
Ieu_13 = (X_update[1]-X_update[4]-X_update[5]*im)/x_13
Ieu_23 = (X_update[2]+X_update[3]*im-X_update[4]-X_update[5]*im)/x_23
EV_update = [X_update[1],X_update[2],X_update[3],real(Ieu_12),imag(Ieu_12),real(Ieu_13),imag(Ieu_13),real(Ieu_23),imag(Ieu_23)]
println("StateEstimation_Vector=[U_1r,U_3r,U_3i,I_12r,I_12i,I_13r,I_13i,I_23r,I_23i] = ",EV_update)
error_update = zeros(n,1)
for i=1:n
    error_update[i]==EV_update[i]-real_value[i]
end
println("新的结果误差 error = ",error_update)
println("可以发现新的结果误差有所减小")
# 👆但是我填不进去😞



1.03089exp(-0.13*im)
1.03614exp(-0.08*im)
1.02362exp(-0.13*im)
1.02737exp(-0.15*im)
1.04633exp(-0.13*im)
1.04886exp(-0.12*im)
1.03324exp(-0.15*im)
1.03003exp(-0.16*im)
1.03005exp(-0.16*im)
1.03796exp(-0.08*im)
1.03998exp(-0.09*im)
1.02365exp(-0.09*im)
1.03284exp(-0.09*im)
1.02507exp(-0.12*im)
1.00777exp(-0.12*im)
1.01772exp(-0.1*im)
1.0144exp(-0.11*im)
1.01566exp(-0.13*im)
1.04352exp(-0.01*im)
0.98742exp(-0.04*im)
1.01803exp(-0.05*im)
1.04046exp(0.03*im)
1.03481exp(0.02*im)
1.02274exp(-0.09*im)
1.04144exp(-0.05*im)
1.01384exp(-0.08*im)
1.00563exp(-0.11*im)
1.01525exp(-0.01*im)
1.02466exp(0.04*im)
(1.03614exp(-0.08*im))
(1.02362exp(-0.13*im))
(1.02737exp(-0.15*im))
(1.04633exp(-0.13*im))
(1.04886exp(-0.12*im))
(1.03003exp(-0.16*im)
(1.03324exp(-0.15*im))
(1.03005exp(-0.16*im))
(1.03796exp(-0.08*im))
(1.02365exp(-0.09*im))
(1.03998exp(-0.09*im))
(1.03284exp(-0.09*im))
(1.02507exp(-0.12*im))
(1.00777exp(-0.12*im))
(1.01772exp(-0.1*im))
(1.0144exp(-0.11*im))
(1.01566exp(-0.13*im))
(1.04352exp(-0.01*im))
(0.98742exp(-0.04*im))
(1.01803exp(-0.05*im))
(1.04046exp(0.03*im))
(1.03481exp(0.02*im))
(1.02274exp(-0.09*im))
(1.04144exp(-0.05*im))
(1.01384exp(-0.08*im))
(1.00563exp(-0.11*im))
(1.01525exp(-0.01*im))
(1.02466exp(0.04*im))


1.03089exp(-7.24*pi/180*im)
1.03089exp(-7.24*pi/180*im)
1.03614exp(-4.67*pi/180*im)
1.03614exp(-4.67*pi/180*im)
1.02362exp(-7.60*pi/180*im)
1.02362exp(-7.60*pi/180*im)
1.02737exp(-8.47*pi/180*im)
1.02737exp(-8.47*pi/180*im)
1.04633exp(-7.40*pi/180*im)
1.04633exp(-7.40*pi/180*im)
1.04886exp(-6.75*pi/180*im)
1.04886exp(-6.75*pi/180*im)
1.03324exp(-8.77*pi/180*im)
1.03324exp(-8.77*pi/180*im)
1.03003exp(-9.24*pi/180*im)
1.03003exp(-9.24*pi/180*im)
1.03005exp(-8.99*pi/180*im)
1.03005exp(-8.99*pi/180*im)
1.03796exp(-4.38*pi/180*im)
1.03796exp(-4.38*pi/180*im)
1.03998exp(-5.20*pi/180*im)
1.03998exp(-5.20*pi/180*im)
1.02365exp(-5.17*pi/180*im)
1.02365exp(-5.17*pi/180*im)
1.03284exp(-5.05*pi/180*im)
1.03284exp(-5.05*pi/180*im)
1.02507exp(-6.63*pi/180*im)
1.02507exp(-6.63*pi/180*im)
1.00777exp(-6.93*pi/180*im)
1.00777exp(-6.93*pi/180*im)
1.01772exp(-5.45*pi/180*im)
1.01772exp(-5.45*pi/180*im)
1.0144exp(-6.45*pi/180*im)
1.0144exp(-6.45*pi/180*im)
1.01566exp(-7.33*pi/180*im)
1.01566exp(-7.33*pi/180*im)
1.04352exp(- 0.77*pi/180*im)
0.98742exp(-2.20*pi/180*im)
0.98742exp(-2.20*pi/180*im)
1.01803exp(-2.98*pi/180*im)
1.01803exp(-2.98*pi/180*im)
1.04046exp(1.56*pi/180*im)
1.04046exp(1.56*pi/180*im)
1.03481exp( 1.36*pi/180*im)
1.03481exp( 1.36*pi/180*im)
1.02274exp(-5.32*pi/180*im)
1.02274exp(-5.32*pi/180*im)
1.04144exp(-3.13*pi/180*im)
1.01384exp(-4.35*pi/180*im)
1.00563exp(-6.54*pi/180*im)
1.01525exp(-0.61*pi/180*im)
1.02466exp(2.26*pi/180*im)
1.04750exp(-2.22*pi/180*im)
complex(0.98200)
0.98310exp(3.45*pi/180*im)
0.99720exp(4.47*pi/180*im)
1.01230exp(3.00*pi/180*im)
1.04930exp(6.56*pi/180*im)
1.06350exp(9.28*pi/180*im)
1.02780exp(3.47*pi/180*im)
1.02650exp(9.43*pi/180*im)
1.03000exp(-8.82*pi/180*im)
real(1.03089exp(-0.08*im))
imag(1.03089exp(-0.08*im))
real(1.03614exp(-0.08*im))
imag(1.03614exp(-0.08*im))
real(1.02362exp(-0.13*im))
imag(1.02362exp(-0.13*im))
real(1.02737exp(-0.15*im))
imag(1.02737exp(-0.15*im))
real(1.04633exp(-0.13*im))
imag(1.04633exp(-0.13*im))
real(1.04886exp(-0.12*im))
imag(1.04886exp(-0.12*im))
real(1.03003exp(-0.16*im))
imag(1.03003exp(-0.16*im))
real(1.03324exp(-0.15*im))
imag(1.03324exp(-0.15*im))
real(1.03005exp(-0.16*im))
imag(1.03005exp(-0.16*im))
real(1.03796exp(-0.08*im))
imag(1.03796exp(-0.08*im))
real(1.02365exp(-0.09*im))
imag(1.02365exp(-0.09*im))
real(1.03998exp(-0.09*im))
imag(1.03998exp(-0.09*im))
real(1.03284exp(-0.09*im))
imag(1.03284exp(-0.09*im))
real(1.02507exp(-0.12*im))
imag(1.02507exp(-0.12*im))
real(1.00777exp(-0.12*im))
imag(1.00777exp(-0.12*im))
real(1.01772exp(-0.1*im))
imag(1.01772exp(-0.1*im))
real(1.0144exp(-0.11*im))
imag(1.0144exp(-0.11*im))
real(1.01566exp(-0.13*im))
imag(1.01566exp(-0.13*im))
real(1.04352exp(-0.01*im))
imag(1.04352exp(-0.01*im))
real(0.98742exp(-0.04*im))
imag(0.98742exp(-0.04*im))
real(1.01803exp(-0.05*im))
imag(1.01803exp(-0.05*im))
real(1.04046exp(0.03*im))
imag(1.04046exp(0.03*im))
real(1.03481exp(0.02*im))
imag(1.03481exp(0.02*im))
real(1.02274exp(-0.09*im))
imag(1.02274exp(-0.09*im))
real(1.04144exp(-0.05*im))
imag(1.04144exp(-0.05*im))
real(1.01384exp(-0.08*im))
imag(1.01384exp(-0.08*im))
real(1.00563exp(-0.11*im))
imag(1.00563exp(-0.11*im))
real(1.01525exp(-0.01*im))
imag(1.01525exp(-0.01*im))
real(1.02466exp(0.04*im))
imag(1.02466exp(0.04*im))
real(1.0475exp(-0.04*im))#1.0475  # bus_gen_below
imag(1.0475exp(-0.04*im))
real(complex(0.98200))
real(0.9831exp(0.06*im))#0.9831
imag(0.9831exp(0.06*im))
real(0.99720exp(0.08*im))
imag(0.99720exp(0.08*im))
real(1.01230exp(0.05*im))#
imag(1.01230exp(0.05*im))
real(1.04930exp(0.11*im))
imag(1.04930exp(0.11*im))
real(1.06350exp(0.16*im))#
imag(1.06350exp(0.16*im))
real(1.02780exp(0.07*im))#
imag(1.02780exp(0.07*im))
real(1.02650exp(0.16*im))#
imag(1.02650exp(0.16*im))
real(1.3exp(-0.15*im))#
imag(1.3exp(-0.15*im))
