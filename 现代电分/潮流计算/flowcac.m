clear ;%清除工作区，即变量清除
clc   %清空命令行窗口
t1=clock;%用于计算程序运行时间
%极坐标形式的牛拉法潮流计算程序
% -----------------------------------------------数据输入模块--------------------------------------------------------------
branch= xlsread('支路数据.xlsx');
br=size(branch);
br=br(1); %br为支路数
node= xlsread('6节点数据.xlsx');
bus=size(node);
bus=bus(1);%bus为节点数
pr=1e-9;
% limit=1e-5
 method=0;%1为PQ分解法
% method=1
%---------------------------------------------导纳矩阵形成模块--------------------------------------------------------------

Y=zeros(bus);
V=zeros(bus,1);
for ii=1:br      %br为支路数
    y=1/(branch(ii,3)+branch(ii,4)*1i);
       if branch(ii,6)==1      %变比为1，无变压器
        p=branch(ii,1);q=branch(ii,2);
        Y(p,q)=Y(p,q)-y;
        Y(q,p)=Y(p,q);
        Y(p,p)=Y(p,p)+y-branch(ii,5)*1i;
        Y(q,q)=Y(q,q)+y-branch(ii,5)*1i;
       else                   %变比不为1，有变压器
        k= branch(ii,6);
        p=branch(ii,1);q=branch(ii,2);
        Y(p,q)=Y(p,q)-y/k;
        Y(q,p)=Y(p,q);
        Y(p,p)=Y(p,p)+y/k^2-branch(ii,5)*1i; 
        Y(q,q)=Y(q,q)+y-branch(ii,5)*1i;
        end
end
%分解导纳的实部虚部
G=real(Y);B=imag(Y);
for ii=1:bus
      B(ii,ii) = B(ii,ii) -node(ii,8);              %B2（i,8)为节点并联电容
end
% theta V电压幅值相位初值
V=ones(bus,1);
theta=zeros(bus,1);
for ii=1:bus    %bus为节点数
    
    V(ii,1)=node(ii,2);   %电压幅值
    theta(ii,1)=node(ii,3); %电压相角
    
end
%迭代次数Times
Times=0;
%求出PQ节点的个数
npq=0;
for ii=1:bus  %bus为节点数
    if node(ii,1)==2
        npq=npq+1;
    end
end
npq
%-------------------------------------------------计算不平衡功率模块---------------------------------------------------------------------
M=ones(2*bus,1);
while max(abs(M))> pr %判断是否满足精度条件
PQ=zeros(npq,1); %记录PQ节点的位置
t=0;
for ii=2:bus
    if node(ii,1)==2
        t=t+1;
        PQ(t,1)=ii;
    end
end
% 形成潮流方程
P=zeros(bus,1);    %bus为节点数
Q=zeros(bus,1);
for i=1:bus
    for j=1:bus
        P(i,1)=V(i)*sum((G(i,:).*cos(theta(i)-theta(:))'+B(i,:).*sin(theta(i)-theta(:))')*V(:));
        Q(i,1)=V(i)*sum((G(i,:).*sin(theta(i)-theta(:))'-B(i,:).*cos(theta(i)-theta(:))')*V(:));
    end
end
%求出功率的不平衡量
dP=zeros(bus-1,1);
dQ=zeros(npq,1);
s=0;
t=0;
for ii=2:bus
    s=s+1;
    dP(s,1)=node(ii,4)-node(ii,6)-P(ii);
    if node(ii,1)==2
        t=t+1;
        dQ(t,1)=node(ii,5)-node(ii,7)-Q(ii);
    end
end
% ---------------------------------------------形成雅各比矩阵模块---------------------------------------------------------------------------
H=zeros(bus,bus);
N=zeros(bus,bus);
K=zeros(bus,bus);
L=zeros(bus,bus);
%求出雅克比矩阵元素的表达式
if method==0
[Jacobi]=Jacobi_NR(bus,node,V,theta,B,G,P,Q);
else
    [Jacobi]=Jacobi_PQ(bus,node,V,B);
end
% ----------------------------------------------------------解修正方程模块---------------------------------------------------------------
A=[dP;dQ];
M=Jacobi\A;
dtheta=zeros(bus-1,1);
for ii=1:bus-1
    dtheta(ii,1)=M(ii);       %得到dtheta
end
for ii=1:bus-1
    theta(ii+1,1)=theta(ii+1)+dtheta(ii); %形成新的theta
end
dVM=zeros(npq,1);
for ii=bus:size(M)
    dVM(ii+1-bus,1)=M(ii);
end
dVM                   %得到dV/V
dV=zeros(npq,1);
dV(PQ,1)=V(PQ,1).*dVM;       %得到dV
for ii=1:bus
    V(ii)=V(ii)+dV(ii);       %形成新的V
end
w=zeros(bus,1);
for ii=1:bus
    w(ii,1)=theta(ii)*180/pi;   %弧度转角度
end
Times=Times+1;
for ii=1:bus
    P(ii,1)=V(ii)*sum((G(ii,:).*cos(theta(ii)-theta(:))'+B(ii,:).*sin(theta(ii)-theta(:))')*V(:));
    Q(ii,1)=V(ii)*sum((G(ii,:).*sin(theta(ii)-theta(:))'-B(ii,:).*cos(theta(ii)-theta(:))')*V(:));
end
 %判断PV节点无功是否越界
for ii=1:bus
    if node(ii,1)~=2
        node(ii,5)=node(ii,7)+Q(ii);    %node(ii,5)为发电机发出的无功，等于负荷无功加上注入无功。等式为书上公式82页
        if node(ii,1)==3             
            if node(ii,5)>=node(ii,9)
                node(ii,5)=node(ii,9);   %发出无功定在上界
                node(ii,1)=2;            %改为PQ节点
            elseif node(ii,5)<=node(ii,10)
                node(ii,5)=node(ii,10);   %发出无功定在下界
                node(ii,1)=2;         
            end
        end
    end
end
disp(['第' num2str(Times) '次迭代：'])
disp('   电压幅值   相角')
disp([V,w])
end
% -----------------------------------------------------计算线路潮流------------------------------------------------------------------
t=0;
for ii=1:bus
    if node(ii,1)==3
        t=t+1;
        Q_PV(t,1)=node(ii,5)-node(ii,7);
       
       
    end
end
P_balance=P(1);       %注入有功
for ii=1:br
        p=branch(ii,1);q=branch(ii,2);
        S_line(ii,1)=p;
        S_line(ii,2)=q;
        U(p,1)=V(p)*(cos(theta(p))+1i*sin(theta(p)));
        U(q,1)=V(q)*(cos(theta(q))+1i*sin(theta(q)));
        S_line(ii,3)=real(U(p,1)*(conj(U(p,1))-conj(U(q,1)))*conj(Y(p,q))+U(p,1)^2*(-branch(ii,5))*1i);
        S_line(ii,5)=real(U(q,1)*(conj(U(q,1))-conj(U(p,1)))*(G(q,p)-1i*B(q,p))+U(q,1)^2*(-branch(ii,5))*1i);
        if branch(ii,6)~=1
            k=branch(ii,6);
        S_line(ii,4)=imag(U(p,1)*(conj(U(p,1))-conj(U(q,1)))*(G(p,q)-1i*B(p,q))+(U(p,1)^2*((-branch(ii,5))*1i)-Y(p,q)*(1-k)/k^2));
        S_line(ii,6)=imag(U(q,1)*(conj(U(q,1))-conj(U(p,1)))*(G(q,p)-1i*B(q,p))+(U(q,1)^2*((-branch(ii,5))*1i)-Y(p,q)*(k-1)/k));
        else
        S_line(ii,4)=imag(U(p,1)*(conj(U(p,1))-conj(U(q,1)))*(G(p,q)-1i*B(p,q))+U(p,1)^2*(-branch(ii,5))*1i);
        S_line(ii,6)=imag(U(q,1)*(conj(U(q,1))-conj(U(p,1)))*(G(q,p)-1i*B(q,p))+U(q,1)^2*(-branch(ii,5))*1i);
         end
        S_line(ii,7)= (S_line(ii,3)^2+S_line(ii,4)^2)/V(p)^2*branch(ii,3);%有功损耗
        S_line(ii,8)= (S_line(ii,3)^2+S_line(ii,4)^2)/V(p)^2*branch(ii,4);%无功损耗
       
end
% ----------------------------------------------------结果输出模块-------------------------------------------------------------------------
T=fix(clock);
disp('---------------------------------------------------输出结果-------------------------------------------------------------------')
% disp('程序运行时间')
% disp(['程序总运行时间：',num2str(etime(clock,t1))]);
disp('节点导纳矩阵：')
disp(Y)
disp('收敛次数times=')
disp(Times)
disp('收敛后：')
disp('节点电压编号 电压幅值  相位')
node1=1:bus;
node1=node1';
disp([node1,V,w])
disp('支路功率及损耗：')
disp('   首端节点  末端节点   首端有功   首端无功  末端有功  末端无功  有功损耗  无功损耗' )
disp(S_line)
disp('PV节点输出无功')
disp(Q_PV)


disp('平衡节点功率：')
disp('   有功功率  无功功率')
disp([P_balance,node(1,5)])

disp('程序运行时间')
disp(['程序总运行时间：',num2str(etime(clock,t1))]);