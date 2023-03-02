% ���ų�����OPF�����ڵ㷨���
% ˵�������������һ�����ļ�main.m�����������ļ�makeY.m��Coeff.m��dPQ.m���ĸ��ļ���Ӧ����MATLAB��ǰĿ¼��
% ����main.m���Ὣ��������ӡ����Ļ���������ļ�solution.txt��
clc;
clear;
close all
tic;
%% ��������
%�ڵ����ݱ�
%      1  2  3    4      5        6        7        8       9       10      11     12      13      14     15      16      17    18    19
%     �ڵ� �� �� ��ѹ     ���    �й�     �޹�     �й�    �޹�   ��ѹ       ����    ��ֵ    ��ֵ    ����   ����    Զ�˿� �ڵ�   ��ѹ  ��ѹ
%      ��  �� ��                  ����     ����     ����    ����   ��׼      ��ѹ    ����    ����    �絼   ����     �ƽڵ� ��     ����  ����
  N=[  1  1  3   1.00   0.00     0.00     0.00     1.10    0.00   230.00   1.00   0.0000  0.0000  0.0000  0.0000    0     1     1.07   0.95
       2  1  2   1.00   0.00     0.00     0.00     0.50    0.00   230.00   1.00   0.0000  0.0000  0.0000  0.0000    0     2     1.07   0.95
       3  1  2   1.00   0.00     0.00     0.00     0.50    0.00   230.00   1.00   0.0000  0.0000  0.0000  0.0000    0     3     1.07   0.95
       4  1  1   1.00   0.00     1.00     0.15     0.00    0.00   230.00   1.00   0.0000  0.0000  0.0000  0.0000    0     4     1.07   0.95
       5  1  1   1.00   0.00     1.00     0.15     0.00    0.00   230.00   1.00   0.0000  0.0000  0.0000  0.0000    0     5     1.07   0.95
       6  1  1   1.00   0.00     1.00     0.15     0.00    0.00   230.00   1.00   0.0000  0.0000  0.0000  0.0000    0     6     1.07   0.95 ];

 %֧·���ݱ�
%          1  2   3  4 5   6   7       8      9       10    11      12     13    14    15       16     17   18   19   20   21   22    
%          �� �� �� �� �� �� ����   �翹   ����  ֧·�� ֧·�� ֧·��  ����  λ ��ѹ���� �������� ��С ��� ���� ��С ��� ֧·  
%                �� �� · ��                     ������ �޹��� �޹��� ĸ�ߺ� �� �մ���� ������� ��� ���      ��ѹ ��ѹ  ��
     B = [ 1  2  1  1  1   0  0.1   0.200    0.04      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    1
           1  4  1  1  1   0  0.05  0.200    0.04      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    2
           1  5  1  1  1   0  0.08  0.300    0.30      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    3
           2  3  1  1  1   0  0.05  0.25     0.25      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    4
           2  4  1  1  1   0  0.05  0.1      0.02      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    5
           2  5  1  1  1   0  0.1   0.3      0.04      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    6
           2  6  1  1  1   0  0.07  0.2      0.05      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    7
           3  5  1  1  1   0  0.12  0.26     0.05      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    8
           3  6  1  1  1   0  0.02  0.1      0.02      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    9
           4  5  1  1  1   0  0.2   0.4      0.08      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    10
           5  6  1  1  1   0  0.1   0.3      0.08      0.    0.    5.      0     0    1.0       0.0    0.0  0.0  0.0  0.0  0.0    11 ];   

%��������ݱ�
%           1    2     3    4     5     6       7         8        9   
%            ���� �ڵ� �й� �޹� �й� �޹�   ����      һ��       ����  
%            ����  ��  �Ͻ� �Ͻ� �½� �½�   ϵ��      ϵ��        ��
  Gen=[     1    1    2    1.5  0.5    -1.0   0.00533  11.669   213.1
            2    2   1.5   1.5  0.375  -1.0   0.00889  10.333   200
            3    3   1.8   1.5  0.45   -1.0   0.00741  10.333   240];

[num_node,~] = size(N);
[num_branch,~] = size(B);

% ��ͬ���ͽڵ����
num_PQ = 0;
num_PV = 0;
for k = 1:num_node
  if (N(k,3) == 1)
      num_PQ = num_PQ+1;
  end
    if (N(k,3) == 2)
      num_PV = num_PV+1;
    end
end

num_gen = num_node-num_PQ; % 3 ���������
num_equa = 2*num_node; %12 ��ʽԼ������m
num_inequa = 2*num_gen+num_node+num_branch; %23 ����ʽԼ������r
num_state = 2*num_gen+2*num_node; %18 ״̬��������n
a2 = Gen(:,7); 
a1 = Gen(:,8); 
a0 = Gen(:,9); %�������Զ���ʽϵ��
A2 = diag(a2); 
A1 = diag(a1);

%% ���ɽڵ㵼�ɾ���
Y = makeY(N, B, num_node, num_branch);

%% ��������
PG = N(1:3, 8);
QR = N(1:3, 9);

P_ejec = N(:,8)-N(:,6); 
Q_ejec = N(:,9)-N(:,7);

Xtilde(1:2:num_node*2-1) = (N(:,5))';
Xtilde(2:2:num_node*2)   = (N(:,4))';

u = ones(num_inequa,1); 
l = ones(num_inequa,1);
z = ones(1,num_inequa); 
w = -0.5*ones(1,num_inequa);
y = 1E-10*ones(1,num_equa); 
y(2:2:num_equa) = -1*y(2:2:num_equa);

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
    x = [PG; QR; Xtilde'];  % ״̬����x
    X = [z'; l; w'; u; x; y']; % ��������X
    g1 = PG; % �й�����g1
    g2 = QR; % �޹�����g2
    g3 = (Xtilde(2:2:num_node*2))'; % �ڵ��ѹ��ֵg3
    
    for k = 1:num_branch % ��·����g4
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
    
    L = diag(l); 
    U = diag(u); 
    Z = diag(z); 
    W = diag(w); 
    % Lagrange����
    % Lagrange������ƫ����
    Ly = h;
    Lz = g-l-g_l;
    Lw = g+u-g_u;
    L_miu_l = L*Z*ones(num_inequa,1)-miu*ones(num_inequa,1);
    L_miu_u = U*W*ones(num_inequa,1)+miu*ones(num_inequa,1);
    
    [A, dh_dx, dg_dx, H_, d2h_dx_y, d2g_dx_c, d2f_dx, temp, L_Z, U_W] = Coeff(num_node, num_PQ, Y, B, X, A2);
    
   %% Ŀ�꺯��
    df_dx = [ 2*A2*PG+a1; zeros(3,1); zeros(length(Xtilde),1)];
    
    %% �������պ���
    Lx = df_dx-dh_dx*y'-dg_dx*(z+w)';
    Lx_ = Lx+dg_dx*(L\(L_miu_l+Z*Lz)+U\(L_miu_u-W*Lw));
    
    b = [-L\L_miu_l; Lz; -U\L_miu_u; -Lw; Lx_; -Ly];
    delta_X = dX(H_, dg_dx, dh_dx, L_Z, U_W, A, b, z, l', u', w, x', y); 
    
    X = X + delta_X;
    
    % ���±���
    z = (X(1:num_inequa))';
    l = X(num_inequa+1:2*num_inequa);
    w = (X(2*num_inequa+1:3*num_inequa))';
    u = X(3*num_inequa+1:4*num_inequa);
    x = X(4*num_inequa+1:4*num_inequa+18);
    y = (X(4*num_inequa+19:122))';

    PG = (x(1:num_gen));
    QR = (x(num_gen+1:2*num_gen));
    P_ejec(1:3) = PG;
    Q_ejec(1:3) = QR;
    Xtilde = (x(2*num_gen+1:2*num_gen+2*num_node))';    
    
end

%% ��Ǳ任��-�е���֮��
for k = 1:num_node
    while Xtilde(2*k-1)>pi
       Xtilde(2*k-1) = Xtilde(2*k-1)-2*pi;
    end
    while Xtilde(2*k-1)<-pi
       Xtilde(2*k-1) = Xtilde(2*k-1)+2*pi;
    end        
end

%% ��5��ƽ��ڵ����Ϊ��׼��������
for k = 1:num_node
    Xtilde(2*k-1) = Xtilde(2*k-1)-Xtilde(2*num_node-1);
end
if (num_iteration>=50)
  disp('OPF is not convergent!');  
end
toc;

% ��������
semilogy(gap_record(1:num_iteration));
grid on;
title('ϵͳ���ų����ڵ㷨��������');
xlabel('��������'); 
ylabel('Gap');

source = [Gen(:,1:2) PG QR A2*(PG.*PG)+A1*PG+a0];
source_sum = sum(source(:,3:5));
voltage = [N(:,1) (Xtilde(2:2:2*num_node))' (Xtilde(1:2:2*num_node-1))'];
branch = [B(:,22) B(:,1:2) g4];

fileID = fopen('solution.txt','w+', 'n', 'UTF-8');
disp( '���ų�����������');
disp('====================================================');
disp( '|                  �й��޹���Դ����                |');
disp('====================================================');
disp( ' ����  �ڵ�    �й�    �޹�     ȼ��');
disp(' ����   ��     ����    ����     ����/$'); 
disp(['  ',num2str(source(1,1)),'     ',num2str(source(1,2)),'    ',num2str(source(1,3)),'   ',num2str(source(1,4)),'    ',num2str(source(1,5))]);
disp(['  ',num2str(source(2,1)),'     ',num2str(source(2,2)),'    ',num2str(source(2,3)),'   ',num2str(source(2,4)),'   ',num2str(source(2,5))]);
disp(['�ܼ�    ��    ',num2str(source_sum(1)),'   ',num2str(source_sum(2)),'   ',num2str(source_sum(3))]);



% disp(['�� �� ֧ ·:              ', num2str(o), '                  ',num2str(a)])

disp('====================================================');
disp( '|                     �ڵ��ѹ����                 |');
disp('====================================================');
disp( '�ڵ�    ��ѹ      ��ѹ');
disp( '��      ��ֵ      ��ǧ�'); 
disp([num2str(voltage(1,1)),'       ',num2str(voltage(1,2)),'      ', num2str(voltage(1,3)) ]);
disp([num2str(voltage(2,1)),'       ',num2str(voltage(2,2)),'      ', num2str(voltage(2,3)) ]);
disp([num2str(voltage(3,1)),'       ',num2str(voltage(3,2)),'   ', num2str(voltage(3,3)) ]);
disp([num2str(voltage(4,1)),'       ',num2str(voltage(4,2)),'   ', num2str(voltage(4,3)) ]);
disp([num2str(voltage(5,1)),'       ',num2str(voltage(5,2)),'        ', num2str(voltage(5,3)) ]);

disp('====================================================');
disp('|                   ֧·�й�����                 |');
disp('====================================================');
disp('֧·  ��  ��   ����');
disp('�� '); 
fprintf(fileID, '\r\n%2d    %2d  %2d  %6.4f', branch');
type('solution.txt');