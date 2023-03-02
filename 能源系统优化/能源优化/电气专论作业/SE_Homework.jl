# Homework for Week 4
# Smart Grid @ BJTU
# By Prof. Yin Xu, 2020-3-15

# LinearAlgebra Package is loaded
using LinearAlgebra

# (1) Calculate True Values
# bus voltage
Vbus = [
    complex(1.0)
    1.02exp(2*pi/180*im)
    1.03exp(3*pi/180*im)
]
# line impedance
Zline = [
    0.01im
    0.02im
    0.015im
]
# line admittance
Yline = 1 ./Zline
# line susceptance
Bline = imag(Yline)
# incidence matrix
A = [
     1  1  0
    -1  0  1
     0 -1 -1
]
# Line currents
Iline = A'*Vbus./Zline

# (2) Measurement Equation
# Measurement matrix
H = zeros(9,5)
    H[1,1] = H[2,4] = H[3,5] = 1
    H[4,3] = H[5,1] = Bline[1]
    H[5,2] = -Bline[1]
    H[6,5] = H[7,1] = Bline[2]
    H[7,4] = -Bline[2]
    H[8,5] = H[9,2] = Bline[3]
    H[8,3] = H[9,4] = -Bline[3]

# Measurements
z = [0.995 1.029 0.545 -2.0 1.9 -2.71 1.0 -1.22 0.598]'

# (3) LS Estimate
Vbus_est_LS = (H'*H)\H'*z
Vbus_est_LS_comp = [
    Vbus_est_LS[1] + 0im
    Vbus_est_LS[2] + Vbus_est_LS[3]*im
    Vbus_est_LS[4] + Vbus_est_LS[5]*im
]
Iline_est_LS = A'*Vbus_est_LS_comp./Zline

# (4) WLS Estimate
W = Matrix(1.0I,9,9)
W[9,9] = 0.01
W[4,4] = 0.01
W[7,7] = 0.01
W[3,3] = 0.01
# Estimate states using WLS
Vbus_est_WLS = (H'*W*H)\H'*W*z
Vbus_est_WLS_comp = [
    Vbus_est_WLS[1] + 0im
    Vbus_est_WLS[2] + Vbus_est_WLS[3]*im
    Vbus_est_WLS[4] + Vbus_est_WLS[5]*im
]
Iline_est_WLS = A'*Vbus_est_WLS_comp./Zline
