%两点乘积算法
clear
clc
f=50;
w=2*pi*f;

t1=0.0321;%采样时间点
i1=5*sqrt(2)*sin(w*t1-pi/6);
t2=t1+0.005;
i2=5*sqrt(2)*sin(w*t2-pi/6);

%计算有效值
I=sqrt((i1*i1+i2*i2)/2);

%计算误差
I_error=((I-5)/5)*100;

%输出显示
fprintf("电流有效值： %2.2f\n",I)
fprintf("  电流误差： %2.2f%%\n",I_error)