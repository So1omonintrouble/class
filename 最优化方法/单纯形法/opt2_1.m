A=[-8 -2;-4 -2;-5,-1];
b=[-5;-4;-2];
Aeq=[];
beq=[];
lb = [0,0];
f=[320 100];
[x,fmin]=linprog(f,A,b,Aeq,beq,lb)