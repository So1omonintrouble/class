% 两点乘积算法：计算电流有效值，电流测量误差百分数
% 两点乘积算法：计算电压除以电流得到的阻抗模值、阻抗角
close all;clc;clear;

f = 50;                % 频率
T = 1/f;               % 周期
N = 12;                % 采样点数
w = 2*pi*f;             
t = (0:T/N:T)';
m = size(t);

U_t= 10*sqrt(2)*sin(w*t+pi/6);            % 基波电压
I_t= 5*sqrt(2)*sin(w*t-pi/6);             % 基波电流

I_t_AAA = (w*t-pi/6)/pi*180;                % 电压、电流实际相角
U_t_AAA = (w*t+pi/6)/pi*180;

%% 两点乘积算法
I_t_phase = 5*sqrt(2)*sin(w.*t-pi/6+pi/2);          % 向右平移pi/2
I_t_amp = sqrt((I_t.^2+I_t_phase.^2)/2);            % 计算电流的有效值
I_t_angle = atand(I_t./I_t_phase);                  % 计算电流两相角差

U_t_phase = 10*sqrt(2)*sin(w*t+pi/6+pi/2);          % 向右平移pi/2
U_t_amp = sqrt((U_t.^2+U_t_phase.^2)/2);            % 计算电压的有效值
U_t_angle = atand(U_t./U_t_phase);                  % 计算电压两相角差

%% 计算阻抗的模和阻抗角
% 对电压电流相角处理
for i = 2:1:length(I_t_angle)
    while I_t_angle(i-1) > I_t_angle(i)
        I_t_angle(i) = I_t_angle(i) + 180;
    end
    while U_t_angle(i-1) > U_t_angle(i)
        U_t_angle(i) = U_t_angle(i) + 180;
    end
end

Z = U_t_amp./I_t_amp;            % 阻抗模值
Z_a = U_t_angle - I_t_angle;     % 阻抗角

X = Z.*sind(Z_a);
R = Z.*cosd(Z_a);

%% 可视化
figure('Name','两点乘积算法') 
set(figure(1),'unit', 'centimeters', 'position', [23, 1, 15, 19]);    
subplot(5,1,1);hold on
plot(t,I_t,'r',t,I_t_phase,'b');
legend('i_t','i_t-T/4');
title('电流移相90°');

subplot(5,1,2);
plot(t,U_t,'r',t,U_t_phase,'b');
legend('U_t','U_t-T/4');
title('电压移相90°');

subplot(5,1,3);
plot(t,I_t_amp,'-or',t,U_t_amp,'-ob');
legend('电流有效值','电压有效值');
axis([0,T,-1.5*max(U_t_amp),1.5*max(U_t_amp)]);
xlabel('t (s)');ylabel('有效值');
title('计算电流有效值、电压有效值');
hold on

subplot(5,1,4);hold on
plot(t,Z,'-k');
plot(t,R,'-r');
plot(t,X,'-b');
legend('阻抗Z','电阻分量R','电抗分量X');
axis([0,T,-1.5*max([max(Z),max(R),max(X)]),1.5*max([max(Z),max(R),max(X)])]);                
xlabel('t (s)');ylabel('数值');
title('两点乘积算法的阻抗模值');

subplot(5,1,5);
plot(t,I_t_angle,'-or',t,U_t_angle,'-ob',t,Z_a,'-ok');
legend('电流相角','电压相角','阻抗角');
axis([0,0.02,-1.2*max([max(I_t_angle),max(U_t_angle)]),1.2*max([max(I_t_angle),max(U_t_angle)])]);     
xlabel('t (s)');ylabel('角度（°）');
title('两点乘积算法的阻抗角');

