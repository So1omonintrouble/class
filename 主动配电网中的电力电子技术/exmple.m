clc;
clear;
close all
tic;
%% 算例数据
%节点数据表
%       1  2  3   4        5       6        7        8       9      10      11     12      13      14     15      16    17    18    19
%     节点 区 类 电压     相角    有功     无功     有功    无功   电压   期望    乏值    乏值    并联   并联   远端控 节点  电压  电压
%      号  号 型                  负荷     负荷     出力    出力   基准   电压    上限    下限    电导   电纳  制节点号 号   上限  下限
%
  N=[  1  1   0 1.0000   0.00     1.60     0.80     0.00    0.00  230.00 1.032   0.0000  0.0000  0.0000  0.0000    0    1    1.1   0.9
        2  1  0 1.0000   0.00     2.00     1.00     0.00    0.00   18.00 1.025   0.0000  0.0000  0.0000  0.0000    0    2     1.1   0.9
        3  1  0 1.0000   0.00     3.70     1.30     0.00    0.00   13.80 1.025   0.0000  0.0000  0.0000  0.0000    0    3    1.1   0.9
        4  1  2 1.0000   0.00     0.00     0.00     4.50    0.00  230.00 1.026   0.0000  0.0000  0.0000  0.0000    0    4     1.1   0.9
        5  1  3 1.0500   0.00     0.00     0.00     4.50    1.45    0.00 0.996   0.0000  0.0000  0.0000  0.0000    0    5     1.1   0.9];
% N=[   1  1  3 1.0700   0.00     0     0.00     1.10       0.00  230.00 1.032   0.0000  0.0000  0.0000  0.0000    0    1    1.07   0.95
%         2  1  2 1.0700   0.00     0     0.00     0.50       0.00  230.00 1.025   1.0000  0.0000  0.0000  0.0000    0    2    1.07   0.95
%         3  1  2 1.0700   0.00     0     0.00     0.50       0.00  230.00 1.025   0.6000  0.0000  0.0000  0.0000    0    3    1.07   0.95
%         4  1  0 1.0000   0.00     1     0.15     0.00       0.00  230.00 1.026   1.5000  -1.0000  0.0000  0.0000    0    4   1.07   0.95
%         5  1  0 1.0000   0.00     1     0.15     0.00       0.00  230.00 0.996   1.5000  -1.0000  0.0000  0.0000    0    5   1.07   0.95
%         6  1  0 1.0000   0.00     1     0.15     0.00       0.00  230.00 1.000   1.2000  -1.0000  0.0000  0.0000    0    6   1.07   0.95];
%支路数据表
%          1  2  3  4  5  6   7      8       9     10    11     12     13    14    15       16     17   18   19   20   21   22    
%          从 到 区 区 电 类 电阻   电抗   电纳  支路额 支路下 支路上  控制  位 变压器最 移相器最 最小 最大 步长 最小 最大 支路  
%                号 号 路 型                     定功率 限功率 限功率 母线号 置 终传输比 终移相角 变比 变比      电压 电压  号
  B=[     1  2  1  1  1  0  0.040  0.2500 0.50      0.    0.    2.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    1
           1  3  1  1  1  0  0.100  0.3500   0.0      0.    0.    0.65    0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    2
           2  3  1  1  1  0  0.080  0.3000 0.50      0.    0.    2.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    3
           2  4  1  1  1  1  0.0    0.0150   0.0      0.    0.    6.      0    0    1.050     0.0    0.0  0.0  0.0  0.0 0.0    4
           3  5  1  1  1  1  0.0    0.0300   0.0      0.    0.    5.      0    0    1.050     0.0    0.0  0.0  0.0  0.0 0.0    5];
%     B=[      1  2  1  1  1  0  0.100  0.2000 0.02      0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    1
%            1  4  1  1  1  0  0.050  0.2000 0.02      0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    2
%            1  5  1  1  1  0  0.080  0.3000 0.03      0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    3
%            2  3  1  1  1  0  0.050  0.2500 0.03      0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    4
%            2  4  1  1  1  0  0.050  0.0200 0.01      0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    5
%            2  5  1  1  1  0  0.100  0.0400 0.02      0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    6
%            2  6  1  1  1  0  0.070  0.0500 0.025     0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    7
%            3  5  1  1  1  0  0.120  0.0500 0.025     0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    8
%            3  6  1  1  1  0  0.020  0.0200 0.01      0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    9
%            4  5  1  1  1  0  0.200  0.0800 0.04      0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    10
%            5  6  1  1  1  0  0.100  0.0600 0.03      0.    0.    0.      0    0    1.0       0.0    0.0  0.0  0.0  0.0 0.0    11
%            ];
%发电机数据表
%             1    2     3    4    5   6       7         8         9   
%            发电 节点 有功 无功 有功 无功   二次      一次       常数  
%            机号  号  上界 上界 下界 下界   系数      系数        项
  Gen=[     1    4    8.0  3.0  1.0  -3.0  50.4395  200.4335  1200.6485
            2     5    8.0  5.0  1.0  -2.1  200.55   500.746   1857.201];
% Gen=[     1    1    1.1  1.5  0.0  -1.0  11.669     213.1     0.00533
%             2    2    0.5  1.5  0.0  -1.0  10.333     200       0.00889
%             3    3    0.5  1.2  0.0  -1.0  10.833     240       0.00741];

[num_node,~] = size(N);
[num_branch,~] = size(B);
% 不同类型节点个数
num_PQ = 0;
num_PV = 0;
for k = 1:num_node
  if (N(k,3) == 0)
      num_PQ = num_PQ+1;
  end
    if (N(k,3) == 2)
      num_PV = num_PV+1;
    end
end

num_gen = num_node-num_PQ; % 2 发电机个数
num_equa = 2*num_node; %10 等式约束个数m
num_inequa = 2*num_gen+num_node+num_branch; %14 不等式约束个数r
num_state = 2*num_gen+2*num_node; % 14 状态变量个数n
Pload = N(1:num_PQ,6); Qload = N(1:num_PQ,7);

a2 = Gen(:,7); a1 = Gen(:,8); a0 = Gen(:,9); %耗量特性多项式系数
A2 = diag(a2); A1 = diag(a1);
%% 生成节点导纳矩阵
Y = makeY(N, B, num_node, num_branch);
%% 参数设置
PG = N(num_PQ+1:num_node, 8);
QR = N(num_PQ+1:num_node, 9);
P_ejec = N(:,8)-N(:,6); Q_ejec = N(:,9)-N(:,7);

Xtilde(1:2:num_node*2-1) = (N(:,5))'; Xtilde(2:2:num_node*2) = (N(:,4))';

u = ones(num_inequa,1); l = ones(num_inequa,1);
z = ones(1,num_inequa); w = -0.5*ones(1,num_inequa); y = 1E-10*ones(1,num_equa); y(2:2:num_equa) = -1*y(2:2:num_equa);

epsi = 1E-6;
sigma = 0.1;
num_iteration = 0; 
max_iteration = 50; 

g_u = [Gen(:,3); Gen(:,4); N(:,18); B(:,12)];
g_l = [Gen(:,5); Gen(:,6); N(:,19); -B(:,12)];
gap_record = zeros(max_iteration,1);
for num_iteration = 1:max_iteration
    gap = l'*z'-u'*w';
    gap_record(num_iteration) = gap;
    if (gap < epsi)
        disp('iterations completed!');
        break; 
    end
  miu = gap*sigma/2/num_inequa;
  x = [PG; QR; Xtilde'];  % 状态变量x
    X = [z'; l; w'; u; x; y']; % 待修正的X
    g1 = PG; g2 = QR; % 有功出力g1，无功出力g2
    g3 = (Xtilde(2:2:num_node*2))'; % 节点电压幅值g3
    for k = 1:num_branch % 线路潮流g4
        ii = B(k,1); jj = B(k,2);
        theta = Xtilde(ii*2-1)-Xtilde(jj*2-1);
        g4(k,1) = Xtilde(2*ii)*Xtilde(2*jj)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta))-Xtilde(2*ii)*Xtilde(2*ii)*real(Y(ii,jj));
    end
    g = [g1; g2; g3; g4];
    
    h = zeros(num_node*2,1);
    for ii = 1:num_node
        for jj = 1:num_node
            theta = Xtilde(ii*2-1)-Xtilde(jj*2-1);
            h(2*ii-1) = h(2*ii-1)-Xtilde(2*ii)*Xtilde(2*jj)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta));
            h(2*ii) = h(2*ii)-Xtilde(2*ii)*Xtilde(2*jj)*(real(Y(ii,jj))*sin(theta)-imag(Y(ii,jj))*cos(theta));
        end
            h(2*ii-1) = h(2*ii-1)+P_ejec(ii);
            h(2*ii) = h(2*ii)+Q_ejec(ii);
    end
    
    
    L = diag(l); U = diag(u); Z = diag(z); W = diag(w); % Lagrange函数
    % Lagrange函数的偏导数
    Ly = h;
    Lz = g-l-g_l;
    Lw = g+u-g_u;
    L_miu_l = L*Z*ones(num_inequa,1)-miu*ones(num_inequa,1);
    L_miu_u = U*W*ones(num_inequa,1)+miu*ones(num_inequa,1);
    
    [A, dh_dx, dg_dx, H_, d2h_dx_y, d2g_dx_c, d2f_dx, temp, L_Z, U_W] = Coeff(num_node, num_PQ, Y, B, X, A2);
    %% 目标函数
    df_dx = [2*A2*PG+a1; zeros(2,1); zeros(length(Xtilde),1)];
    %% 拉格朗日函数
    Lx = df_dx-dh_dx*y'-dg_dx*(z+w)';
    Lx_ = Lx+dg_dx*(L\(L_miu_l+Z*Lz)+U\(L_miu_u-W*Lw));
    
    b = [-L\L_miu_l; Lz; -U\L_miu_u; -Lw; Lx_; -Ly];
    delta_X = dX(H_, dg_dx, dh_dx, L_Z, U_W, A, b, z, l', u', w, x', y); 
    
    X = X + delta_X;
    
    % 更新变量
    z = (X(1:num_inequa))';
    l = X(num_inequa+1:2*num_inequa);
    w = (X(2*num_inequa+1:3*num_inequa))';
    u = X(3*num_inequa+1:4*num_inequa);
    x = X(4*num_inequa+1:5*num_inequa);
    y = (X(5*num_inequa+1:5*num_inequa+num_equa))';

    PG = (x(1:num_gen));
    QR = (x(num_gen+1:2*num_gen));
    P_ejec(num_PQ+1:num_node) = PG;
    Q_ejec(num_PQ+1:num_node) = QR;
    Xtilde = (x(2*num_gen+1:2*num_gen+2*num_node))';    
end
%相角变换到-π到π之间
for k = 1:num_node
    while Xtilde(2*k-1)>pi
       Xtilde(2*k-1) = Xtilde(2*k-1)-2*pi;
    end
    while Xtilde(2*k-1)<-pi
       Xtilde(2*k-1) = Xtilde(2*k-1)+2*pi;
    end        
end
%以5号平衡节点相角为基准进行折算
for k = 1:num_node
    Xtilde(2*k-1) = Xtilde(2*k-1)-Xtilde(2*num_node-1);
end
if (num_iteration>=50)
  disp('OPF is not convergent!');  
end
toc;
% 保存数据
semilogy(gap_record(1:num_iteration));
grid on;
title('系统最优潮流内点法收敛特性');
xlabel('迭代次数'); ylabel('Gap');

source = [Gen(:,1:2) PG QR A2*(PG.*PG)+A1*PG+a0];
source_sum = sum(source(:,3:5));
voltage = [N(:,1) (Xtilde(2:2:2*num_node))' (Xtilde(1:2:2*num_node-1))'];
branch = [B(:,22) B(:,1:2) g4];

fileID = fopen('solution.txt','w+', 'n', 'UTF-8');
disp( '最优潮流计算结果：');
disp('====================================================');
disp( '|                  有功无功电源出力                |');
disp('====================================================');
disp( ' 发电  节点    有功    无功     燃料');
disp(' 机号   号     出力    出力     费用/$'); 
disp(['  ',num2str(source(1,1)),'     ',num2str(source(1,2)),'    ',num2str(source(1,3)),'   ',num2str(source(1,4)),'    ',num2str(source(1,5))]);
disp(['  ',num2str(source(2,1)),'     ',num2str(source(2,2)),'    ',num2str(source(2,3)),'   ',num2str(source(2,4)),'   ',num2str(source(2,5))]);
disp(['总计    —   ',num2str(source_sum(1)),'   ',num2str(source_sum(2)),'   ',num2str(source_sum(3))]);



% disp(['断 开 支 路:              ', num2str(o), '                  ',num2str(a)])

disp('====================================================');
disp( '|                     节点电压相量                 |');
disp('====================================================');
disp( '节点    电压      电压');
disp( '号      幅值      相角'); 
disp([num2str(voltage(1,1)),'       ',num2str(voltage(1,2)),'      ', num2str(voltage(1,3)) ]);
disp([num2str(voltage(2,1)),'       ',num2str(voltage(2,2)),'      ', num2str(voltage(2,3)) ]);
disp([num2str(voltage(3,1)),'       ',num2str(voltage(3,2)),'   ', num2str(voltage(3,3)) ]);
disp([num2str(voltage(4,1)),'       ',num2str(voltage(4,2)),'   ', num2str(voltage(4,3)) ]);
disp([num2str(voltage(5,1)),'       ',num2str(voltage(5,2)),'        ', num2str(voltage(5,3)) ]);

disp('====================================================');
disp('|                    支路有功功率                  |');
disp('====================================================');
disp('支路  从  到   功率');
disp('号 '); 
fprintf(fileID, '\r\n%2d    %2d  %2d  %6.4f', branch');
type('solution.txt');