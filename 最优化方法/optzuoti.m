clear
clc
x0=[1;1;1;1];%初值
ess=1e-2;%精度要求
%定义方程，梯度，海森矩阵
syms x1 x2 x3 x4
x=[x1;x2;x3;x4];
A=[5 1 0 0.5
    1 4 0.5 0
    0 0.5 3 0
    0.5 0 0 2];
f= 0.5 *x'*x+0.25*(x'*A*x)^2;
df=[diff(f,x1);diff(f,x2);diff(f,x3);diff(f,x4)];
Hesse=hessian(f,x);
%迭代求解
k=0;
x_k=x0;
fx_k=subs(f,x,x0);
dfx_k=subs(df,x,x0);
Hessex_k=subs(Hesse,x,x0);
while norm(dfx_k)>ess && k<1
    dx=-Hessex_k^(-1)*dfx_k;
    x_k=x_k+dx;
    fx_k=subs(f,x,x_k);
    dfx_k=subs(df,x,x_k);
    Hessex_k=subs(Hesse,x,x_k);
    k=k+1;
end
%输出小数点后四位
x_k=vpa(x_k,4)
fx_k=vpa(fx_k,4)
dfx_k=vpa(dfx_k,4)
Hessex_k=vpa(Hessex_k,4)
