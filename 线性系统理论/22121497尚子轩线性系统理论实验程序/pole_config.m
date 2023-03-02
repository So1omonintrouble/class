clear
clc
% 状态空间矩阵
A=[ 0 1  0   0
    0 0  0   0
    0 0  0   1
    0 0 29.4 0];
B=[0 1 0 3]';
C=[ 1 0 0 0
    0 0 1 0];
D=[0 0]';

% 目标极点
lamda1=-10;
lamda2=-10;
lamda3=-2+2*sqrt(3)*1i;
lamda4=-2-2*sqrt(3)*1i;
[~,Lambda]=eig(A);
disp('计算特征值为');
disp(Lambda);

% 根据特征值判断稳定性
for i=1:4
    if real(Lambda(i,i))<0
        i=i+1;
    else
       stable=0;
       disp('稳定系数stable（1为稳定，0为不稳定）=') 
       disp(stable) % stable=0 means unstable
       break
    end
end

% 极点配置
if stable==0 % unstable
   % System controllability is determined prior to pole configuration
    Qc=[B,A*B,A*A*B,A*A*A*B];
    disp('能控性判别矩阵Qc=');
    disp(Qc);
    disp('rank(Qc)=');
    disp(rank(Qc));
    if rank(Qc)==4
       control=1;
    else
       control=0;
    end
    disp('能控系数control（1为能控，0为不能控）=')
    disp(control)  

    lamda=diag([lamda1,lamda2,lamda3,lamda4]);
    alpha=poly(A);
    alpha1=poly(lamda);
    Qcc=[A*A*A*B A*A*B A*B B];
    QC=[   1       0        0      0;
          alpha(2)   1        0      0;
          alpha(3) alpha(2)   1      0;
          alpha(4) alpha(3) alpha(2) 1];
    P=Qcc*QC;
    k=[alpha1(5)-alpha(5) alpha1(4)-alpha(4) alpha1(3)-alpha(3) alpha1(2)-alpha(2)]/P;
    disp('状态反馈k=')
    disp(k)
      
    % 引入状态反馈后的A矩阵（用A1表示）
    A1=A-B*k;
    sim('actual_operation')
     
    % 时域仿真结果展示
    figure(1) % before
    plot(x0(:,1),x0(:,2),phi0(:,1),phi0(:,2))
    legend('\fontname{宋体}位置\fontname{times new roman}x_0','\fontname{宋体}角度\fontname{times new roman}\phi_0')
    set(gcf,'Position',[700,50,600,300]);
    figure(2) % after
    plot(x1(:,1),x1(:,2),phi1(:,1),phi1(:,2))
    legend('\fontname{宋体}位置\fontname{times new roman}x_1','\fontname{宋体}角度\fontname{times new roman}\phi_1')
    set(gcf,'Position',[700,50,600,300]);
else
    disp('无需极点配置')
end