A=[-1 -1 0 1 1 0 0 1 0
    1 0 0 -1 0 0 1 0 1
    0 1 -1 0 -1 1 0 0 -1];
E0=[1 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0];
Yb=diag([1 1 1 1 1 1 1 1 1]);
Yn=A*Yb*A';
Y=E0*Yb*E0'-E0*Yb*A'*Yn^(-1)*A*Yb*E0'