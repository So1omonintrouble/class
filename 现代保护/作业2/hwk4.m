%半周差分积分算法
clear
clc
f=50;
w=2*pi*f;
Ts=5/3*1e-3;

t0=0.00032;%采样时间点
S=0;
i_Sigma=0;
for i=1:12/2
    t=t0+Ts*i;
    i_Sigma=i_Sigma+abs(5*sqrt(2)*sin(w*t-pi/6)-5*sqrt(2)*sin(w*(t-Ts)-pi/6));
end
%计算有效值
I=i_Sigma/2/sqrt(2);

%计算误差
I_error=((I-5)/5)*100;

%输出显示
fprintf("电流有效值： %2.2f\n",I)
fprintf("  电流误差：%2.2f%%\n",I_error)