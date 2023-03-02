A=[1 2;3 2;0,1];
b=[6;12;2];
Aeq=[];
beq=[];
lb = [0,0];
f=[-3 -4];
[x,fmax]=linprog(f,A,b,Aeq,beq,lb)