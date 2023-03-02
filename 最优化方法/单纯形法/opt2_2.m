A=[-1 -2 -3;-2 -2 -1];
b=[-5;-6];
Aeq=[];
beq=[];
lb = [0,0,0];
f=[3 4 5];
[x,fmin]=linprog(f,A,b,Aeq,beq,lb)