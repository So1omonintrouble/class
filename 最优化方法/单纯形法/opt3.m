A=[1 3 1;4 7 1];
b=[6;15];
Aeq=[];
beq=[];
lb = [0,0,0];
f=[-3 -5 -1];
[x,fmax]=linprog(f,A,b,Aeq,beq,lb)