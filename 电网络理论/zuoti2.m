Yb=[1	0	0	0	0	0	0
0	1i	0	0	0	0	0
0	0	-1i	0	0	0	0
0	0	0	1i	0	0	0
0	0	0	0	-1i	0	0
0	0	0	0	0	1	0
2	0	0	0	0	0	1
];
A=[0	-1	0	0	0	1	1
0	0	1	0	0	-1	0
0	0	-1	1	0	0	0
1	0	0	-1	-1	0	0
0	1	0	0	1	0	0
];
Yn=A*Yb*A';
Aw=[0 0 1 0 -1
    0 1 0 0 0];
Aw'