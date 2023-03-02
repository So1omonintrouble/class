%导数算法
clear
clc
f=50;
w=2*pi*f;
Ts=5/3*1e-3;

t1=0.02132;%采样时间点
i1=5*sqrt(2)*sin(w*t1-pi/6);
t2=t1+Ts;
di1=1/Ts*(5*sqrt(2)*sin(w*t2-pi/6)-i1);

%计算有效值
I=sqrt((i1^2+(di1/w)^2)/2);

%计算误差
I_error=((I-5)/5)*100;

%输出显示
fprintf("电流有效值： %2.2f\n",I)
fprintf("  电流误差：%2.2f%%\n",I_error)