%两点乘积算法计算阻抗模值阻抗角
clear
clc
f=50;
w=2*pi*f;
t0=0.00032;

i1=5*sqrt(2)*sin(w*t0-pi/6);
u1=10*sqrt(2)*sin(w*t0+pi/6);
t1=t0+5e-3;
i2=5*sqrt(2)*sin(w*t1-pi/6);
u2=10*sqrt(2)*sin(w*t1+pi/6);

Z=sqrt((u1^2+u2^2)/(i1^2+i2^2));
alpha_Z=atan(u1/u2)-atan(i1/i2);

fprintf("阻抗模量： %2.2f\n",Z)
fprintf("  阻抗角：%2.2f\n",rad2deg(alpha_Z))
