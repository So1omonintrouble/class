A=[1 0;0 1;3,2];
b=[4;6;18];
Aeq=[];
beq=[];
lb = [0,0];
f=[-3 -5];
[x,fmax]=linprog(f,A,b,Aeq,beq,lb)