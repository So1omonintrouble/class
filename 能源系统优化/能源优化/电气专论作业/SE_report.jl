# Report for stateEstimate
# Smart Grid @ BJTU
# By Prof. Yin Xu, 2020-3-15
# Revised by Stu. Yanran Du, 2020-4-10

# LinearAlgebra Package is loaded
using LinearAlgebra

# (1) Calculate True Values
# bus voltage
Vbus = [
        1.03089exp(-7.24*pi/180*im)
        1.03614exp(-4.67*pi/180*im)
        1.02400exp(-7.45*pi/180*im)
        1.02737exp(-8.47*pi/180*im)
        1.04633exp(-7.40*pi/180*im)
        1.04886exp(-6.75*pi/180*im)
        1.03324exp(-8.77*pi/180*im)
        1.03003exp(-9.24*pi/180*im)
        1.03005exp(-8.99*pi/180*im)
        1.03796exp(-4.38*pi/180*im)
        1.03998exp(-5.20*pi/180*im)
        1.02365exp(-5.17*pi/180*im)
        1.03284exp(-5.05*pi/180*im)
        1.02507exp(-6.63*pi/180*im)
        1.00777exp(-6.93*pi/180*im)
        1.01772exp(-5.45*pi/180*im)
        1.01440exp(-6.45*pi/180*im)
        1.01566exp(-7.33*pi/180*im)
        1.04352exp(-0.77*pi/180*im)
        0.98742exp(-2.20*pi/180*im)
        1.01803exp(-2.98*pi/180*im)
        1.04046exp(1.56*pi/180*im)
        1.03481exp(1.36*pi/180*im)
        1.02274exp(-5.32*pi/180*im)
        1.04144exp(-3.13*pi/180*im)
        1.01384exp(-4.35*pi/180*im)
        1.00563exp(-6.54*pi/180*im)
        1.01525exp(-0.57*pi/180*im)
        1.02466exp(2.26*pi/180*im)
        1.04750exp(-2.22*pi/180*im)
        complex(0.98200)
        0.98310exp(3.45*pi/180*im)
        0.99720exp(4.47*pi/180*im)
        1.01230exp(3.00*pi/180*im)
        1.04930exp(6.56*pi/180*im)
        1.06350exp(9.28*pi/180*im)
        1.02780exp(3.47*pi/180*im)
        1.02650exp(9.43*pi/180*im)
        1.03000exp(-8.82*pi/180*im)
]
# line impedance
Zline = [
    #      R	     X
        0.00350 +	0.04110  *im
        0.00100 +	0.02500  *im
        0.00130 +	0.01510  *im
        0.00700 +	0.00860  *im
        0.00000 +	0.01810  *im
        0.00110 +	0.01330  *im
        0.00130 +	0.02130  *im
        0.00080 +	0.01290  *im
        0.00080 +	0.01280  *im
        0.00080 +	0.01120  *im
        0.00020 +	0.00260  *im
        0.00070 +	0.00820  *im
        0.00060 +	0.00920  *im
        0.00040 +	0.00460  *im
        0.00230 +	0.03630  *im
        0.00100 +	0.02500  *im
        0.00000 +	0.02000  *im
        0.00040 +	0.00430  *im
        0.00040 +	0.00430  *im
        0.00160 +	0.04350  *im
        0.00160 +	0.04350  *im
        0.00090 +	0.01010  *im
        0.00180 +	0.02170  *im
        0.00090 +	0.00940  *im
        0.00030 +	0.00590  *im
        0.00080 +	0.01350  *im
        0.00160 +	0.01950  *im
        0.00070 +	0.00890  *im
        0.00130 +	0.01730  *im
        0.00070 +	0.00820  *im
        0.00070 +	0.01420  *im
        0.00070 +	0.01380  *im
        0.00090 +	0.01800  *im
        0.00080 +	0.01400  *im
        0.00000 +	0.01430  *im
        0.00060 +	0.00960  *im
        0.00050 +	0.02720  *im
        0.00220 +	0.03500  *im
        0.00060 +	0.02320  *im
        0.00320 +	0.03230  *im
        0.00570 +	0.06250  *im
        0.00430 +	0.04740  *im
        0.00140 +	0.01470  *im
        0.00140 +	0.01510  *im
        0.00080 +	0.01560  *im
        0.00000 +	0.02500  *im
]
# line admittance
Yline = 1 ./Zline
# line susceptance
Bline = imag(Yline)
Gline = real(Yline)
# incidence matrix
branch_relation = [
                    1	2
                    1	39
                    2	3
                    2	25
                    2	30
                    3	18
                    3	4
                    4	14
                    4	5
                    5	8
                    5	6
                    6	11
                    6	7
                    7	8
                    8	9
                    9	39
                    10	32
                    10	13
                    10	11
                    12	13
                    12	11
                    13	14
                    14	15
                    15	16
                    16	24
                    16	21
                    16	19
                    16	17
                    17	27
                    17	18
                    19	33
                    19	20
                    20	34
                    21	22
                    22	35
                    22	23
                    23	36
                    23	24
                    25	37
                    25	26
                    26	29
                    26	28
                    26	27
                    28	29
                    29	38
                    31	6
]
n = 39
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
H = zeros(2*50-1,2n-1)   #99 77
    for i = 1:2n-1
        H[i,i] = 1
    end
 #   H[i,1*n]
        H[78,39*2-1] = H[79,1*2-1] = Bline[2]
        H[78,1*2] = H[79,39*2-2] = -Bline[2] #1
        H[78,1*2-1] = H[79,1*2] = Gline[2]
        H[78,39*2-2] = H[79,39*2-1] = -Gline[2] #1

        H[80,30*2] = H[81,2*2-1] = Bline[5] #2
        H[80,2*2] = H[81,30*2-1] = -Bline[5]
        H[80,2*2-1] = H[81,2*2] = Gline[5] #2
        H[80,30*2-1] = H[81,30*2] = -Gline[5]

        H[82,39*2-1] = H[83,9*2-1] = Bline[16] #3
        H[82,9*2] = H[83,39*2-2] = -Bline[16]
        H[82,9*2-1] = H[83,9*2] = Gline[16] #3
        H[82,39*2-2] = H[83,39*2-1] = -Gline[16]

        H[84,32*2-1] = H[85,10*2-1] = Bline[17] #4
        H[84,10*2] = H[85,32*2-2] = -Bline[17]
        H[84,10*2-1] = H[85,10*2] = Gline[17] #4
        H[84,32*2-2] = H[85,32*2-1] = -Gline[17]

        H[86,33*2-1] = H[87,19*2-1] = Bline[31] #5
        H[86,19*2] = H[87,33*2-2] = -Bline[31]
        H[86,19*2-1] = H[87,19*2] = Gline[31] #5
        H[86,33*2-2] = H[87,33*2-1] = -Gline[31]

        H[88,34*2-1] = H[89,20*2-1] = Bline[33] #6
        H[88,20*2] = H[89,34*2-2] = -Bline[33]
        H[88,20*2-1] = H[89,20*2] = Gline[33] #6
        H[88,34*2-2] = H[89,34*2-1] = -Gline[33]

        H[90,35*2-1] = H[91,22*2-1] = Bline[35] #7
        H[90,22*2] = H[91,35*2-2] = -Bline[35]
        H[90,22*2-1] = H[91,22*2] = Gline[35] #7
        H[90,35*2-2] = H[91,35*2-1] = -Gline[35]

        H[92,36*2-1] = H[93,23*2-1] = Bline[37] #8
        H[92,23*2] = H[93,36*2-2] = -Bline[37]
        H[92,23*2-1] = H[93,23*2] = Gline[37] #8
        H[92,36*2-2] = H[93,36*2-1] = -Gline[37]

        H[94,37*2-1] = H[95,25*2-1] = Bline[39] #9
        H[94,25*2] = H[95,37*2-2] = -Bline[39]
        H[94,25*2-1] = H[95,25*2] = Gline[39] #9
        H[94,37*2-2] = H[95,37*2-1] = -Gline[39]

        H[96,38*2-1] = H[97,29*2-1] = Bline[45] #10
        H[96,29*2] = H[97,38*2-2] = -Bline[45]
        H[96,29*2-1] = H[97,29*2] = Gline[45] #10
        H[96,38*2-2] = H[97,38*2-1] = -Gline[45]

        H[98,6*2] = H[99,31*2-2] = Bline[46] #11
        H[99,6*2-1] = -Bline[46]
        H[98,31*2-2] = Gline[46] #11
        H[98,6*2-1] = H[99,6*2] = -Gline[46]

# Measurements
z = [ # 在10个发电机节点处添加PMU测量得到发电机节点处的电压以及与之相连11条支路的电流 由于需要估计39个节点的电压向量值，数据冗余度不够故再添加其他29个节点（图个省事虽然实际浪费测量仪器，但先做一下能不能算吧，而且后面的没改数据）共100-1个测量数据
      # 刚刚发现 ctrl+/ 可以整体注释，便于测试；
      # 4-9  为了验证程序正确性采取简单的方式设置测量值
      # 4-10 从结果看初步验证为正确，如需减少节点电压测量增加电流测量可自此基础上改进
      # 4-10 又有问题了，我都没动啊，怎么就
      # 以后只要正确就记得保存副本。
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
    imag(Vbus[7])
    real(Vbus[8])
    imag(Vbus[8])
    real(Vbus[9])
    imag(Vbus[9])
    real(Vbus[10])
    imag(Vbus[10])
    real(Vbus[11])
    imag(Vbus[11])
    real(Vbus[12])
    imag(Vbus[12])
    real(Vbus[13])
    imag(Vbus[13])
    real(Vbus[14])
    imag(Vbus[14])
    real(Vbus[15])
    imag(Vbus[15])
    real(Vbus[16])
    imag(Vbus[16])
    real(Vbus[17])
    imag(Vbus[17])
    real(Vbus[18])
    imag(Vbus[18])
    real(Vbus[19])
    imag(Vbus[19])
    real(Vbus[20])
    imag(Vbus[20])
    real(Vbus[21])
    imag(Vbus[21])
    real(Vbus[22])
    imag(Vbus[22])
    real(Vbus[23])
    imag(Vbus[23])
    real(Vbus[24])
    imag(Vbus[24])
    real(Vbus[25])
    imag(Vbus[25])
    real(Vbus[26])
    imag(Vbus[26])
    real(Vbus[27])
    imag(Vbus[27])
    real(Vbus[28])
    imag(Vbus[28])
    real(Vbus[29])
    imag(Vbus[29])
    real(Vbus[30])
    real(Vbus[31])
    imag(Vbus[31])
    real(Vbus[32])
    imag(Vbus[32])
    real(Vbus[33])
    imag(Vbus[33])
    real(Vbus[34])
    imag(Vbus[34])
    real(Vbus[35])
    imag(Vbus[35])
    real(Vbus[36])
    imag(Vbus[36])
    real(Vbus[37])
    imag(Vbus[37])
    real(Vbus[38])
    imag(Vbus[38])
    real(Vbus[39])
    imag(Vbus[39])
    real(Iline[2])
    imag(Iline[2])
    real(Iline[5])
    imag(Iline[5])
    real(Iline[16])
    imag(Iline[16])
    real(Iline[17])
    imag(Iline[17])
    real(Iline[31])
    imag(Iline[31])
    real(Iline[33])
    imag(Iline[33])
    real(Iline[35])
    imag(Iline[35])
    real(Iline[37])
    imag(Iline[37])
    real(Iline[39])
    imag(Iline[39])
    real(Iline[45])
    imag(Iline[45])
    real(Iline[46])
    imag(Iline[46])
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
    Vbus_est_LS[13] + Vbus_est_LS[14]*im
    Vbus_est_LS[15] + Vbus_est_LS[16]*im
    Vbus_est_LS[17] + Vbus_est_LS[18]*im
    Vbus_est_LS[19] + Vbus_est_LS[20]*im
    Vbus_est_LS[21] + Vbus_est_LS[22]*im
    Vbus_est_LS[23] + Vbus_est_LS[24]*im
    Vbus_est_LS[25] + Vbus_est_LS[26]*im
    Vbus_est_LS[27] + Vbus_est_LS[28]*im
    Vbus_est_LS[29] + Vbus_est_LS[30]*im
    Vbus_est_LS[31] + Vbus_est_LS[32]*im
    Vbus_est_LS[33] + Vbus_est_LS[34]*im
    Vbus_est_LS[35] + Vbus_est_LS[36]*im
    Vbus_est_LS[37] + Vbus_est_LS[38]*im
    Vbus_est_LS[39] + Vbus_est_LS[40]*im
    Vbus_est_LS[41] + Vbus_est_LS[42]*im
    Vbus_est_LS[43] + Vbus_est_LS[44]*im
    Vbus_est_LS[45] + Vbus_est_LS[46]*im
    Vbus_est_LS[47] + Vbus_est_LS[48]*im
    Vbus_est_LS[49] + Vbus_est_LS[50]*im
    Vbus_est_LS[51] + Vbus_est_LS[52]*im
    Vbus_est_LS[53] + Vbus_est_LS[54]*im
    Vbus_est_LS[55] + Vbus_est_LS[56]*im
    Vbus_est_LS[57] + Vbus_est_LS[58]*im
    Vbus_est_LS[59] + 0*im
    Vbus_est_LS[60] + Vbus_est_LS[61]*im
    Vbus_est_LS[62] + Vbus_est_LS[63]*im
    Vbus_est_LS[64] + Vbus_est_LS[65]*im
    Vbus_est_LS[66] + Vbus_est_LS[67]*im
    Vbus_est_LS[68] + Vbus_est_LS[69]*im
    Vbus_est_LS[70] + Vbus_est_LS[71]*im
    Vbus_est_LS[72] + Vbus_est_LS[73]*im
    Vbus_est_LS[74] + Vbus_est_LS[75]*im
    Vbus_est_LS[76] + Vbus_est_LS[77]*im
]
Iline_est_LS = A'*Vbus_est_LS_comp./Zline

# (4) WLS Estimate
W = Matrix(1.0I,99,99)
 W[1,1] = 0.001
# Estimate states using WLS
Vbus_est_WLS = (H'*W*H)\H'*W*z
Vbus_est_WLS_comp = [
    Vbus_est_WLS[1] + Vbus_est_WLS[2]*im
    Vbus_est_WLS[3] + Vbus_est_WLS[4]*im
    Vbus_est_WLS[5] + Vbus_est_WLS[6]*im
    Vbus_est_WLS[7] + Vbus_est_WLS[8]*im
    Vbus_est_WLS[9] + Vbus_est_WLS[10]*im
    Vbus_est_WLS[11] + Vbus_est_WLS[12]*im
    Vbus_est_WLS[13] + Vbus_est_WLS[14]*im
    Vbus_est_WLS[15] + Vbus_est_WLS[16]*im
    Vbus_est_WLS[17] + Vbus_est_WLS[18]*im
    Vbus_est_WLS[19] + Vbus_est_WLS[20]*im
    Vbus_est_WLS[21] + Vbus_est_WLS[22]*im
    Vbus_est_WLS[23] + Vbus_est_WLS[24]*im
    Vbus_est_WLS[25] + Vbus_est_WLS[26]*im
    Vbus_est_WLS[27] + Vbus_est_WLS[28]*im
    Vbus_est_WLS[29] + Vbus_est_WLS[30]*im
    Vbus_est_WLS[31] + Vbus_est_WLS[32]*im
    Vbus_est_WLS[33] + Vbus_est_WLS[34]*im
    Vbus_est_WLS[35] + Vbus_est_WLS[36]*im
    Vbus_est_WLS[37] + Vbus_est_WLS[38]*im
    Vbus_est_WLS[39] + Vbus_est_WLS[40]*im
    Vbus_est_WLS[41] + Vbus_est_WLS[42]*im
    Vbus_est_WLS[43] + Vbus_est_WLS[44]*im
    Vbus_est_WLS[45] + Vbus_est_WLS[46]*im
    Vbus_est_WLS[47] + Vbus_est_WLS[48]*im
    Vbus_est_WLS[49] + Vbus_est_WLS[50]*im
    Vbus_est_WLS[51] + Vbus_est_WLS[52]*im
    Vbus_est_WLS[53] + Vbus_est_WLS[54]*im
    Vbus_est_WLS[55] + Vbus_est_WLS[56]*im
    Vbus_est_WLS[57] + Vbus_est_WLS[58]*im
    Vbus_est_WLS[59] + 0*im
    Vbus_est_WLS[60] + Vbus_est_WLS[61]*im
    Vbus_est_WLS[62] + Vbus_est_WLS[63]*im
    Vbus_est_WLS[64] + Vbus_est_WLS[65]*im
    Vbus_est_WLS[66] + Vbus_est_WLS[67]*im
    Vbus_est_WLS[68] + Vbus_est_WLS[69]*im
    Vbus_est_WLS[70] + Vbus_est_WLS[71]*im
    Vbus_est_WLS[72] + Vbus_est_WLS[73]*im
    Vbus_est_WLS[74] + Vbus_est_WLS[75]*im
    Vbus_est_WLS[76] + Vbus_est_WLS[77]*im
]
Iline_est_WLS = A'*Vbus_est_WLS_comp./Zline

# difference
E = [
    Vbus[1]-Vbus_est_WLS_comp[1]
    Vbus[2]-Vbus_est_WLS_comp[2]
    Vbus[3]-Vbus_est_WLS_comp[3]
    Vbus[4]-Vbus_est_WLS_comp[4]
    Vbus[5]-Vbus_est_WLS_comp[5]
    Vbus[6]-Vbus_est_WLS_comp[6]
    Vbus[7]-Vbus_est_WLS_comp[7]
    Vbus[8]-Vbus_est_WLS_comp[8]
    Vbus[9]-Vbus_est_WLS_comp[9]
    Vbus[10]-Vbus_est_WLS_comp[10]
    Vbus[11]-Vbus_est_WLS_comp[11]
    Vbus[12]-Vbus_est_WLS_comp[12]
    Vbus[13]-Vbus_est_WLS_comp[13]
    Vbus[14]-Vbus_est_WLS_comp[14]
    Vbus[15]-Vbus_est_WLS_comp[15]
    Vbus[16]-Vbus_est_WLS_comp[16]
    Vbus[17]-Vbus_est_WLS_comp[17]
    Vbus[18]-Vbus_est_WLS_comp[18]
    Vbus[19]-Vbus_est_WLS_comp[19]
    Vbus[20]-Vbus_est_WLS_comp[20]
    Vbus[21]-Vbus_est_WLS_comp[21]
    Vbus[22]-Vbus_est_WLS_comp[22]
    Vbus[23]-Vbus_est_WLS_comp[23]
    Vbus[24]-Vbus_est_WLS_comp[24]
    Vbus[25]-Vbus_est_WLS_comp[25]
    Vbus[26]-Vbus_est_WLS_comp[26]
    Vbus[27]-Vbus_est_WLS_comp[27]
    Vbus[28]-Vbus_est_WLS_comp[28]
    Vbus[29]-Vbus_est_WLS_comp[29]
    Vbus[30]-Vbus_est_WLS_comp[30]
    Vbus[31]-Vbus_est_WLS_comp[31]
    Vbus[32]-Vbus_est_WLS_comp[32]
    Vbus[33]-Vbus_est_WLS_comp[33]
    Vbus[34]-Vbus_est_WLS_comp[34]
    Vbus[35]-Vbus_est_WLS_comp[35]
    Vbus[36]-Vbus_est_WLS_comp[36]
    Vbus[37]-Vbus_est_WLS_comp[37]
    Vbus[38]-Vbus_est_WLS_comp[38]
    Vbus[39]-Vbus_est_WLS_comp[39]
]
println(E)
