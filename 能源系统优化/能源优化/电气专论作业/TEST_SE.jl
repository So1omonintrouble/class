# Homework for Week 4
# Smart Grid @ BJTU
# By Prof. Yin Xu, 2020-3-15

# LinearAlgebra Package is loaded
using LinearAlgebra

# (1) Calculate True Values
# bus voltage
Vbus = [
        144.900exp(12.27*pi/180*im)
        143.520exp(9.26*pi/180*im)
        137.777exp(8.00*pi/180*im)
        138.000exp(9.75*pi/180*im)
        140.054exp(1.62*pi/180*im)
        143.520exp(5.09*pi/180*im)
        complex(143.520)
]
# line impedance
Zline = [
        0.00500+0.05000*im
        0.02000+0.24000*im
        0.01500+0.18000*im
        0.01500+0.18000*im
        0.01000+0.12000*im
        0.00500+0.06000*im
        0.00250+0.03000*im
        0.02000+0.24000*im
        0.00500+0.06000*im
        0.02000+0.24000*im
        0.02000+0.24000*im
]
# line admittance
Yline = 1 ./Zline
# line susceptance
Bline = imag(Yline)
Gline = real(Yline)
# incidence matrix
branch_relation = [
                    1	2
                    1	3
                    2	3
                    2	4
                    2	5
                    2	6
                    3	4
                    4	5
                    7	5
                    6	7
                    6	7
]
n = 7
(m,nons)=size(branch_relation)
A=zeros(n,m)
    for i = 1:n
        for j = 1:m
            if branch_relation[j,1] == i
                A[i,j] = 1   # 流出为正
            end
            if branch_relation[j,2] == i
                A[i,j] = -1  # 流入为负
            end
        end
    end
# Line currents
Iline = A'*Vbus./Zline

# (2) Measurement Equation
# Measurement matrix
# 对所有节点电压都测量13+7条支路14
H = zeros(27,2n-1)
    for i = 1:2n-1
        H[i,i] = 1
    end
    #   H[i,1*n]
        H[14,2*2] = H[15,1*2-1] = Bline[1]#1
        H[14,1*2] = H[15,2*2-1] = -Bline[1]
        H[14,1*2-1] = H[15,1*2] = Gline[1]
        H[14,2*2-1] = H[15,2*2] = -Gline[1]

        H[16,3*2] = H[17,1*2-1] = Bline[2]#1
        H[16,1*2] = H[17,3*2-1] = -Bline[2]
        H[16,1*2-1] = H[17,1*2] = Gline[2]
        H[16,3*2-1] = H[17,3*2] = -Gline[2]

        H[18,3*2] = H[19,2*2-1] = Bline[3]#1
        H[18,2*2] = H[19,3*2-1] = -Bline[3]
        H[18,2*2-1] = H[19,2*2] = Gline[3]
        H[18,3*2-1] = H[19,3*2] = -Gline[3]

        H[20,4*2] = H[21,2*2-1] = Bline[4]#1
        H[20,2*2] = H[21,4*2-1] = -Bline[4]
        H[20,2*2-1] = H[21,2*2] = Gline[4]
        H[20,4*2-1] = H[21,4*2] = -Gline[4]

        H[20,5*2] = H[21,2*2-1] = Bline[5]#1
        H[20,2*2] = H[21,5*2-1] = -Bline[5]
        H[20,2*2-1] = H[21,2*2] = Gline[5]
        H[20,5*2-1] = H[21,5*2] = -Gline[5]

        H[20,6*2] = H[21,2*2-1] = Bline[6]#1
        H[20,2*2] = H[21,6*2-1] = -Bline[6]
        H[20,2*2-1] = H[21,2*2] = Gline[6]
        H[20,6*2-1] = H[21,6*2] = -Gline[6]

        H[20,4*2] = H[21,3*2-1] = Bline[7]#1
        H[20,3*2] = H[21,4*2-1] = -Bline[7]
        H[20,3*2-1] = H[21,3*2] = Gline[7]
        H[20,4*2-1] = H[21,4*2] = -Gline[7]

        H[20,5*2] = H[21,4*2-1] = Bline[8]#1
        H[20,4*2] = H[21,5*2-1] = -Bline[8]
        H[20,4*2-1] = H[21,4*2] = Gline[8]
        H[20,5*2-1] = H[21,5*2] = -Gline[8]

# Measurements
z = [
    real(Vbus[1])
    imag(Vbus[1])
    real(Vbus[2])
    imag(Vbus[2])
    real(Vbus[3])
    imag(Vbus[3])
    real(Vbus[4])
    imag(Vbus[4])
    real(Vbus[5])
    imag(Vbus[5])
    real(Vbus[6])
    imag(Vbus[6])
    real(Vbus[7])
    real(Iline[1])
    imag(Iline[1])
    real(Iline[2])
    imag(Iline[2])
    real(Iline[3])
    imag(Iline[3])
    real(Iline[4])
    imag(Iline[4])
    real(Iline[5])
    imag(Iline[5])
    real(Iline[6])
    imag(Iline[6])
    real(Iline[7])
    imag(Iline[7])
]

# (3) LS Estimate
Vbus_est_LS = (H'*H)\H'*z
Vbus_est_LS_comp = [
    Vbus_est_LS[1] + Vbus_est_LS[2]*im
    Vbus_est_LS[3] + Vbus_est_LS[4]*im
    Vbus_est_LS[5] + Vbus_est_LS[6]*im
    Vbus_est_LS[7] + Vbus_est_LS[8]*im
    Vbus_est_LS[9] + Vbus_est_LS[10]*im
    Vbus_est_LS[11] + Vbus_est_LS[12]*im
    Vbus_est_LS[13] + 0*im

]
Iline_est_LS = A'*Vbus_est_LS_comp./Zline

# (4) WLS Estimate
W = Matrix(1.0I,27,27)
W[4,4] = 0.01
W[7,7] = 0.01
# Estimate states using WLS
Vbus_est_WLS = (H'*W*H)\H'*W*z
Vbus_est_WLS_comp = [
    Vbus_est_WLS[1] + Vbus_est_WLS[2]*im
    Vbus_est_WLS[3] + Vbus_est_WLS[4]*im
    Vbus_est_WLS[5] + Vbus_est_WLS[6]*im
    Vbus_est_WLS[7] + Vbus_est_WLS[8]*im
    Vbus_est_WLS[9] + Vbus_est_WLS[10]*im
    Vbus_est_WLS[11] + Vbus_est_WLS[12]*im
    Vbus_est_WLS[13] + 0*im
]
Iline_est_WLS = A'*Vbus_est_WLS_comp./Zline
