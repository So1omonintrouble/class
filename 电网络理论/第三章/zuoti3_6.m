clear
clc
Baa=[1	0	0	0	1	-1	1	-1	1	0	0	0	0
0	1	0	0	0	0	1	-1	1	0	0	0	0
0	0	1	0	0	0	0	0	0	1	-1	0	0
0	0	0	1	0	0	0	0	0	0	1	1	1
];
Bba=[0	0	0	0	0	1	0	1	0	-1	0	-1	0
0	0	0	0	0	0	0	1	0	0	0	-1	0
];
Bbb=[1 0 1
    0 1 1];
Zba=diag([1,1,1,1,1,1,1,1,1,1,1,1,1]);
Zbb=diag([1,1,1]);
Ibsa=[0 0 0 1 0 0 0 0 0 0 0 0 0]';
Zla=Baa*Zba*Baa';
Zlab=Baa*Zba*Bba';
Zlba=Bba*Zba*Baa';
Zlb=Bba*Zba*Bba'+Bbb*Zbb*Bbb';
Ela=Baa*Zba*Ibsa;
Elb=Bba*Zba*Ibsa;
Z=[6	3	0	0	-2	-1
3	4	0	0	-1	-1
0	0	3	-1	-1	0
0	0	-1	4	-1	-1
-2	-1	-1	-1	6	3
-1	-1	0	-1	3	4
];
E=[0
0
0
1
0
0
];
I=Z^(-1)*E;
B=[1	0	0	0	1	-1	1	-1	1	0	0	0	0	0	0	0
0	1	0	0	0	0	1	-1	1	0	0	0	0	0	0	0
0	0	1	0	0	0	0	0	0	1	-1	0	0	0	0	0
0	0	0	1	0	0	0	0	0	0	1	1	1	0	0	0
0	0	0	0	0	1	0	1	0	-1	0	-1	0	1	0	1
0	0	0	0	0	0	0	1	0	0	0	-1	0	0	1	1
];
i=B'*I

