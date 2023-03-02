%半周积分算法
clear
clc
f=50;
w=2*pi*f;
Ts=5/3*1e-3;

t0=0.01331;%采样时间点
S=0;
for i=0:12/2
    t=t0+Ts*i;
    if mod(i,12/2)==0
        S=S+abs(5*sqrt(2)*sin(w*t-pi/6))/2*Ts;
    else
        S=S+abs(5*sqrt(2)*sin(w*t-pi/6))*Ts;
    end
end
%计算有效值
I=S*w/2/sqrt(2);

%计算误差
I_error=((I-5)/5)*100;

%输出显示
fprintf("电流有效值： %2.2f\n",I)
fprintf("  电流误差：%2.2f%%\n",I_error)