clc;
clearvars;
close all;
%% 表达式

syms x1 x2;

%f1(x1,x2) = x1*((2*0.0195/1.09*x2+2/1.09/59))-(sqrt((251/1.09/59)^2-x2^2)+6.9/1.09*x2); %临界电压表达式
%f2(x1,x2) = x2*((736*59/x1^2-0.0195*59))-1; %临界负荷系数表达式
f1(x1,x2) = x1*((2*0.0122/0.682*x2+2/0.682/85))-(sqrt((251/0.682/85)^2-x2^2)+4.318/0.682*x2);
f2(x1,x2) = x2*((460*85/x1^2-0.0122*85))-1;

%% 求 Jacobi 矩阵

F(x1,x2) = [f1;f2];
f1_x1_Pd(x1,x2) = diff(f1,x1);
f1_x2_Pd(x1,x2) = diff(f1,x2);
f2_x1_Pd(x1,x2) = diff(f2,x1);
f2_x2_Pd(x1,x2) = diff(f2,x2);
J(x1,x2)=[f1_x1_Pd,f1_x2_Pd;f2_x1_Pd,f2_x2_Pd];

%% 迭代求解

n = 10;%迭代次数
Eps = 1e-5;% 设置迭代的精度
x_start = [150;1];% 迭代初始点

x_k_solve = zeros(2,n+1);
x_k_solve(:,1) = x_start;
dk = zeros(2,n);

for i = 1:n
    dk(:,i) =inv(J(x_k_solve(1,i),x_k_solve(2,i)))*(-F(x_k_solve(1,i),x_k_solve(2,i)));
    if(sqrt(sum(dk(:,i).^2)) < Eps)
        x_k_solve(:,i+1) = x_k_solve(:,i) + dk(:,i);
    break;
    else
        x_k_solve(:,i+1) = x_k_solve(:,i) + dk(:,i);
    end
end 

x_k_solve