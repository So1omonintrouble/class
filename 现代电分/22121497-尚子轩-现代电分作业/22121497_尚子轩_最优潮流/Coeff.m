function [A, dh_dx, dg_dx, H_, d2h_dx_y, d2g_dx_c, d2f_dx, temp, L_Z, U_W] = Coeff(num_node, num_PQ, Y, B, X, A2)
[num_branch, ~]  = size(B);
len_x = 2*(num_node-num_PQ+num_node); % 14
num_gen = num_node-num_PQ; % 2
num_equa = 2*num_node;
num_inequa = 2*num_gen+num_node+num_branch;

z = X(1:num_inequa); 
l = X(num_inequa+1:2*num_inequa);
w = X(2*num_inequa+1:3*num_inequa);
u = X(3*num_inequa+1:4*num_inequa);
x = X(4*num_inequa+1:4*num_inequa+len_x); 
y = X(4*num_inequa+len_x+1:4*num_inequa+len_x+num_equa);
Xtilde = x(2*num_gen+1:len_x);

%% 计算等式约束Jacobian矩阵

dh_dPG = zeros(2,2*num_node);
dh_dQR = zeros(2,2*num_node);

for k = num_PQ+1:num_node
   dh_dPG(k-num_PQ,k*2-1) = 1;
   dh_dQR(k-num_PQ,k*2) = 1;
end

dh_dXtilde = zeros(2*num_node,2*num_node);
for ii = 1:num_node
    for jj =1:num_node
        if(ii ~= jj)
            theta = Xtilde(ii*2-1)-Xtilde(jj*2-1);
            dh_dXtilde(jj*2-1,ii*2-1) = -Xtilde(ii*2)*Xtilde(jj*2)*(real(Y(ii,jj))*sin(theta)-imag(Y(ii,jj))*cos(theta)); %Hij
            dh_dXtilde(jj*2-1,ii*2) = Xtilde(ii*2)*Xtilde(jj*2)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta)); %Kij
            dh_dXtilde(jj*2,ii*2-1) = -Xtilde(ii*2)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta)); %Nij
            dh_dXtilde(jj*2,ii*2) = -Xtilde(ii*2)*(real(Y(ii,jj))*sin(theta)-imag(Y(ii,jj))*cos(theta)); %Lij

            dh_dXtilde(ii*2-1,ii*2-1) = dh_dXtilde(ii*2-1,ii*2-1)+Xtilde(ii*2)*Xtilde(jj*2)*(real(Y(ii,jj))*sin(theta)-imag(Y(ii,jj))*cos(theta)); %Hii
            dh_dXtilde(ii*2-1,ii*2) = dh_dXtilde(ii*2-1,ii*2)-Xtilde(ii*2)*Xtilde(jj*2)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta)); %Kii
            dh_dXtilde(ii*2,ii*2-1) = dh_dXtilde(ii*2,ii*2-1)-Xtilde(jj*2)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta)); %Nii
            dh_dXtilde(ii*2,ii*2) = dh_dXtilde(ii*2,ii*2)-Xtilde(jj*2)*(real(Y(ii,jj))*sin(theta)-imag(Y(ii,jj))*cos(theta)); %Lii

        end
    end
    dh_dXtilde(ii*2,ii*2-1) = dh_dXtilde(ii*2,ii*2-1)-2*Xtilde(ii*2)*real(Y(ii,ii)); %Nij
    dh_dXtilde(ii*2,ii*2) = dh_dXtilde(ii*2,ii*2)+2*Xtilde(ii*2)*imag(Y(ii,ii)); %Lij
    
end

dh_dx = [dh_dPG; dh_dQR; dh_dXtilde];
clear dh_dPG dh_dQR dh_dXtilde;
%Slack节点对应的Jacobian元素为0
%% 计算不等式约束Jacobian矩阵

dg1_dPG = eye(num_gen); 
dg1_dQR = zeros(num_gen,num_gen); 
dg1_dXtilde = zeros(2*num_node,num_gen);

dg2_dPG = zeros(num_gen,num_gen); 
dg2_dQR = eye(num_gen); 
dg2_dXtilde = zeros(2*num_node,num_gen);

dg3_dPG = zeros(num_gen,num_node); 
dg3_dQR = zeros(num_gen,num_node);
dg3_dXtilde = zeros(2*num_node,num_node); 

dg4_dPG = zeros(num_gen,num_branch); 
dg4_dQR = zeros(num_gen,num_branch);
dg4_dXtilde = zeros(2*num_node,num_branch);

for ii = 1:num_node
    dg3_dXtilde(2*ii,ii) = 1;
end

for k = 1:num_branch
   ii = B(k,1); 
   jj = B(k,2);
   theta = Xtilde(ii*2-1)-Xtilde(jj*2-1);
   dg4_dXtilde(2*ii-1,k) = -Xtilde(2*ii)*Xtilde(2*jj)*(real(Y(ii,jj))*sin(theta)-imag(Y(ii,jj))*cos(theta));
   dg4_dXtilde(2*jj-1,k) = Xtilde(2*ii)*Xtilde(2*jj)*(real(Y(ii,jj))*sin(theta)-imag(Y(ii,jj))*cos(theta));
   dg4_dXtilde(2*ii,k) = Xtilde(2*jj)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta))-2*Xtilde(2*ii)*real(Y(ii,jj));
   dg4_dXtilde(2*jj,k) = Xtilde(2*ii)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta));
end

dg_dx = [dg1_dPG dg2_dPG dg3_dPG dg4_dPG;
         dg1_dQR dg2_dQR dg3_dQR dg4_dQR;
         dg1_dXtilde dg2_dXtilde dg3_dXtilde dg4_dXtilde];
     
%clear dg1_dPG dg2_dPG dg3_dPG dg4_dPG dg2_dQR dg3_dQR dg4_dQR dg1_dXtilde dg2_dXtilde dg3_dXtilde dg4_dxtilde;

%% 计算对角矩阵
L_Z = diag(z./l);
U_W = diag(w./u);

%% 计算Hessian矩阵
%计算目标函数的Hessian矩阵
d2f_dx = zeros(len_x,len_x);
d2f_dx(1:num_gen,1:num_gen) = 2*A2;

%计算等式约束的Hessian矩阵与Lagrange乘子y乘积
d2h_dx_y = zeros(len_x,len_x);
a = zeros(2*num_node,2*num_node);
for ii = 1:num_node
    for jj = 1:num_node
        theta = Xtilde(ii*2-1)-Xtilde(jj*2-1);  
        if(jj ~= ii)
           %以下三项需要累加
            a(2*ii-1,2*ii-1) = a(2*ii-1,2*ii-1)+Xtilde(2*ii)*Xtilde(2*jj)*(real(Y(ii,jj))*(cos(theta)*y(2*ii-1)+sin(theta)*y(2*ii)+cos(theta)*y(2*jj-1)-sin(theta)*y(2*jj))...
                +imag(Y(ii,jj))*(sin(theta)*y(2*ii-1)-cos(theta)*y(2*ii)-sin(theta)*y(2*jj-1)-cos(theta)*y(2*jj)));
            a(2*ii-1,2*ii) = a(2*ii-1,2*ii)+Xtilde(2*jj)*(real(Y(ii,jj))*(sin(theta)*y(2*ii-1)-cos(theta)*y(2*ii)+sin(theta)*y(2*jj-1)+cos(theta)*y(2*jj))...
                +imag(Y(ii,jj))*(-cos(theta)*y(2*ii-1)-sin(theta)*y(2*ii)+cos(theta)*y(2*jj-1)-sin(theta)*y(2*jj)));
            a(2*ii,2*ii-1) = a(2*ii,2*ii-1)+Xtilde(2*jj)*(real(Y(ii,jj))*(sin(theta)*y(2*ii-1)-cos(theta)*y(2*ii)+sin(theta)*y(2*jj-1)+cos(theta)*y(2*jj))...
                +imag(Y(ii,jj))*(-cos(theta)*y(2*ii-1)-sin(theta)*y(2*ii)+cos(theta)*y(2*jj-1)-sin(theta)*y(2*jj)));
            
            %以下直接计算不需累加
            a(2*ii-1,2*jj-1) = Xtilde(2*ii)*Xtilde(2*jj)*(real(Y(ii,jj))*(-cos(theta)*y(2*ii-1)-sin(theta)*y(2*ii)-cos(theta)*y(2*jj-1)+sin(theta)*y(2*jj))...
                +imag(Y(ii,jj))*(-sin(theta)*y(2*ii-1)+cos(theta)*y(2*ii)+sin(theta)*y(2*jj-1)+cos(theta)*y(2*jj)));
            a(2*ii-1,2*jj) = Xtilde(2*ii)*(real(Y(ii,jj))*(sin(theta)*y(2*ii-1)-cos(theta)*y(2*ii)+sin(theta)*y(2*jj-1)+cos(theta)*y(2*jj))...
                +imag(Y(ii,jj))*(-cos(theta)*y(2*ii-1)-sin(theta)*y(2*ii)+cos(theta)*y(2*jj-1)-sin(theta)*y(2*jj)));        
            a(2*ii,2*jj-1) = Xtilde(2*jj)*(real(Y(ii,jj))*(-sin(theta)*y(2*ii-1)+cos(theta)*y(2*ii)-sin(theta)*y(2*jj-1)-cos(theta)*y(2*jj))...
                +imag(Y(ii,jj))*(cos(theta)*y(2*ii-1)+sin(theta)*y(2*ii)-cos(theta)*y(2*jj-1)+sin(theta)*y(2*jj)));
            a(2*ii,2*jj) = -(real(Y(ii,jj))*(cos(theta)*y(2*ii-1)+sin(theta)*y(2*ii)+cos(theta)*y(2*jj-1)-sin(theta)*y(2*jj))...
                +imag(Y(ii,jj))*(sin(theta)*y(2*ii-1)-cos(theta)*y(2*ii)-sin(theta)*y(2*jj-1)-cos(theta)*y(2*jj)));
        end

    end
    a(2*ii,2*ii) = -2*(real(Y(ii,ii))*y(2*ii-1)-imag(Y(ii,ii))*y(2*ii));
end
d2h_dx_y(2*num_gen+1:len_x,2*num_gen+1:len_x) = a;
d2g_dx_c = zeros(len_x,len_x);
d2g4_d2Xtilde = zeros(length(Xtilde),length(Xtilde));
c = z+w;

for k = 1:num_branch
   ii = B(k,1); jj = B(k,2);
   theta = Xtilde(ii*2-1)-Xtilde(jj*2-1);
   d2g4_d2Xtilde(2*ii-1,2*ii-1) = d2g4_d2Xtilde(2*ii-1,2*ii-1)+(-Xtilde(2*ii)*Xtilde(2*jj)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta)))*c(2+2+5+k);
   d2g4_d2Xtilde(2*ii-1,2*ii) = d2g4_d2Xtilde(2*ii-1,2*ii)+(-Xtilde(2*jj)*(real(Y(ii,jj))*sin(theta)-imag(Y(ii,jj))*cos(theta)))*c(2+2+5+k);
   d2g4_d2Xtilde(2*ii-1,2*jj-1) = d2g4_d2Xtilde(2*ii-1,2*jj-1)+(Xtilde(2*ii)*Xtilde(2*jj)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta)))*c(2+2+5+k);
   d2g4_d2Xtilde(2*ii-1,2*jj) = d2g4_d2Xtilde(2*ii-1,2*jj)+(-Xtilde(2*ii)*(real(Y(ii,jj))*sin(theta)-imag(Y(ii,jj))*cos(theta)))*c(2+2+5+k);
   d2g4_d2Xtilde(2*jj-1,2*ii-1) = d2g4_d2Xtilde(2*ii-1,2*jj-1);
   d2g4_d2Xtilde(2*jj-1,2*ii) = d2g4_d2Xtilde(2*jj-1,2*ii)+(Xtilde(2*jj)*(real(Y(ii,jj))*sin(theta)-imag(Y(ii,jj))*cos(theta)))*c(2+2+5+k);
   %d2g4_d2Xtilde(2*jj-1,2*jj-1) =  d2g4_d2Xtilde(2*ii-1,2*ii-1);
   d2g4_d2Xtilde(2*jj-1,2*jj-1) = d2g4_d2Xtilde(2*jj-1,2*jj-1)+(-Xtilde(2*ii)*Xtilde(2*jj)*(real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta)))*c(2+2+5+k);

   d2g4_d2Xtilde(2*jj-1,2*jj) = d2g4_d2Xtilde(2*jj-1,2*jj)+(Xtilde(2*ii)*(real(Y(ii,jj))*sin(theta))-imag(Y(ii,jj))*cos(theta))*c(2+2+5+k);
   d2g4_d2Xtilde(2*ii,2*ii-1) = d2g4_d2Xtilde(2*ii-1,2*ii);
   %d2g4_d2Xtilde(2*ii,2*ii) = 0;
   d2g4_d2Xtilde(2*ii,2*ii) = d2g4_d2Xtilde(2*ii,2*ii) - 2*real(Y(ii,jj))*c(2+2+5+k);
   d2g4_d2Xtilde(2*ii,2*jj-1) = d2g4_d2Xtilde(2*jj-1,2*ii);
   d2g4_d2Xtilde(2*ii,2*jj) = (real(Y(ii,jj))*cos(theta)+imag(Y(ii,jj))*sin(theta))*c(2+2+5+k);
   d2g4_d2Xtilde(2*jj,2*ii-1) = d2g4_d2Xtilde(2*ii-1,2*jj);
   d2g4_d2Xtilde(2*jj,2*ii) = d2g4_d2Xtilde(2*ii,2*jj);
   d2g4_d2Xtilde(2*jj,2*jj-1) = d2g4_d2Xtilde(2*jj-1,2*jj);
   d2g4_d2Xtilde(2*jj,2*jj) = 0;
end
d2g_dx_c(2*num_gen+1:len_x,2*num_gen+1:len_x) = d2g4_d2Xtilde;
H_ = -d2f_dx+d2h_dx_y+d2g_dx_c-dg_dx*(L_Z-U_W)*dg_dx';
temp = dg_dx*(L_Z-U_W)*dg_dx';

%% 生成系数矩阵
A = zeros(4*num_inequa+num_equa+len_x,4*num_inequa+num_equa+len_x);
A(1:4*num_inequa,1:4*num_inequa) = eye(4*num_inequa);
A(1:num_inequa,num_inequa+1:2*num_inequa) = L_Z;
A(num_inequa+1:2*num_inequa,4*num_inequa+1:4*num_inequa+len_x) = -dg_dx';
A(2*num_inequa+1:3*num_inequa,3*num_inequa+1:4*num_inequa) = U_W;
A(3*num_inequa+1:4*num_inequa,4*num_inequa+1:4*num_inequa+len_x) = dg_dx';
A(4*num_inequa+1:4*num_inequa+len_x,4*num_inequa+1:4*num_inequa+num_equa+len_x) = [H_ dh_dx];
A(4*num_inequa+len_x+1:4*num_inequa+num_equa+len_x,4*num_inequa+1:4*num_inequa+len_x) = dh_dx';