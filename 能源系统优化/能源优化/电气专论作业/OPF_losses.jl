# OPF_LOSSES , 2019-3-31, By Stu. Yanran Du.
#    for the "Tutorial of Electric Engineering" course @ BJTU
# Homework for Week 5

using LinearAlgebra, Convex, MosekTools
# 设置基值
    Sbase = 1000 #MVA
    Vbase = 4.16/sqrt(3) #kV
    Zbase = Vbase^2*1e3/Sbase # Ω
# 标准线路阻抗数据,实际值ohms per mile
    z601=[0.3465+1.0179im   0.1560+0.5017im   0.1580+0.4236im;
          0.1560+0.5017im   0.3375+1.0478im   0.1535+0.3849im; 0.1580+5.4236im   0.1535+0.3849im   0.3414+1.0348im]
    z602=[0.7526+1.1814im   0.1580+0.4236im   0.1560+0.5017im;
          0.1580+0.4236im   0.7475+1.1983im   0.1535+0.3849im;
          0.1560+0.5017im   0.1535+0.3849im   0.7436+1.2112im]
    z603=[0.0000+0.0000im   0.0000+0.0000im   0.0000+0.0000im;
          0.0000+0.0000im   1.3294+1.3471im   0.2066+0.4591im;
          0.0000+0.0000im   0.2066+0.4591im   1.3238+1.3569im]
    z604=[1.3238+1.3569im   0.0000+0.0000im   0.2066+0.4591im;
          0.0000+0.0000im   0.0000+0.0000im   0.0000+0.0000im;
          0.2066+0.4591im   0.0000+0.0000im   1.3294+1.3471im]
    z605=[0.0000+0.0000im   0.0000+0.0000im   0.0000+0.0000im;
          0.0000+0.0000im   0.0000+0.0000im   0.0000+0.0000im;
          0.0000+0.0000im   0.0000+0.0000im   1.3292+1.3475im]
    z606=[0.7982+0.4463im   0.3192+0.0328im   0.2849-0.0143im;
          0.3192+0.0328im   0.7891+0.4041im   0.3192+0.0328im;
          0.2849-0.0143im   0.3192+0.0328im   0.7982+0.4463im]
    z607=[1.3425+0.5124im   0.0000+0.0000im   0.0000+0.0000im;
          0.0000+0.0000im   0.0000+0.0000im   0.0000+0.0000im;
          0.0000+0.0000im   0.0000+0.0000im   0.0000+0.0000im]
    z608=[38.072+69.222im   0.0000+0.0000im   0.0000+0.0000im;
          0.0000+0.0000im   38.072+69.222im   0.0000+0.0000im;
          0.0000+0.0000im   0.0000+0.0000im   38.072+69.222im]#XFM_1
# 本系统节点负荷信息
    bus=[#node Pₐ      Qₐ      Pᵦ      Qᵦ      P𝒸      Q𝒸     weight
            1	0	    0	    0	    0	    0	    0	    1;
            2	8.5	    5	    33      19	    58.5	34	    1;
            3	0	    0	    0	    0   	0	    0	    1;
            4	160	    110     120     90  	120 	90	    1;
            5	434.91  349.57  418     239     572.09  280.433 1;
            6	485	    190	    68      60	    290	    212	    1;
            7	0	    0	    0	    0   	0	    0	    1;
            8	0	    0	    170     125   	0	    0	    1;
            9	0	    0	    153.11 -0.4  	76.89	132.4   1;
            10	0	    0	    0	    0   	0	    0	    1;
            11	128	    86	    0	    0   	0	    0	    1;
            12	0	    0	    0	    0       170	    80	    1]
# 本系统分布式电源信息
    gen=[# node   Pₘ    Qₘ
            1     0     0  ;
            2     0     0  ;
            3     0     0  ;
            4     500   400;
            5     500   400;
            6     0     0  ;
            7     0     0  ;
            8     0     0  ;
            9     0     0  ;
            10    0     0  ;
            11    560   550;
            12    0     0  ]
# 本系统并联电容器信息(相当于固定发出量的电源)
    Shunt_C=[# node   Qₐ    Qᵦ    Q𝒸
                1     0     0     0  ;
                2     0     0     0  ;
                3     0     0     0  ;
                4     0     0     0  ;
                5     0     0     0  ;
                6     200   200   200;
                7     0     0     0  ;
                8     0     0     0  ;
                9     0     0     0  ;
                10    0     0     0  ;
                11    0     0     100;
                12    0     0     0  ]
# 本系统支路信息
    branch=[#B     E     ft      type
             1     2     2000    601;
             2     8     500     603;
             2     3     500     602;
             2     5     2000    601;
             3     4     0.01    608; #XFM_1
             5     10    300     604;
             5     6     500     606;
             5     7     1000    601;
             8     9     300     603;
             10    12    800     607;
             10    12    300     605]
# 提取本系统相关信息
    (n,nons) = size(bus) # 节点个数
    (m,nons) = size(branch) # 支路个数
    (n_gen,nons) = size(gen) # 电源个数
    sₗ = hcat(bus[:,2]+bus[:,3]*im,bus[:,4]+bus[:,5]*im,bus[:,6]+bus[:,7]*im) # 负载ABC相分别消耗的功率
    #sₗ = bus[:,2]+bus[:,4]+bus[:,6]+(bus[:,3]+bus[:,5]+bus[:,7])*im
    sᵢₘ = gen[:,2]+gen[:,3]*im
    sᵢ𝒸 = hcat(Shunt_C[:,2]*im,Shunt_C[:,3]*im,Shunt_C[:,4])
    c_1 = 0.6 # rmb/kWh
    c_2 = 0.8 # rmb/kWh
    ϕ = 0.85 # 分布式电源功率因素
    O = zeros(ComplexF64,3) # 以防约束中的比较
# 将本系统线路阻抗值与标准信息匹配
    Z=zeros(ComplexF64,3,3,m)
        for i=1:11
            zy=branch[:,4]
            if zy[i]==601
                Z[:,:,i]=z601/Zbase*branch[i,3]/5280
            end
            if zy[i]==602
                Z[:,:,i]=z602/Zbase*branch[i,3]/5280
            end
            if zy[i]==603
                Z[:,:,i]=z603/Zbase*branch[i,3]/5280
            end
            if zy[i]==604
                Z[:,:,i]=z604/Zbase*branch[i,3]/5280
            end
            if zy[i]==605
                Z[:,:,i]=z605/Zbase*branch[i,3]/5280
            end
            if zy[i]==606
                Z[:,:,i]=z606/Zbase*branch[i,3]/5280
            end
            if zy[i]==607
                Z[:,:,i]=z607/Zbase*branch[i,3]/5280
            end
            if zy[i]==608
                Z[:,:,i]=z607/Zbase*branch[i,3]/5280
            end
        end

# 建立模型
    # 定义变量，如何定义类似Z的变量❓❓❓❓❓❓❓❓❓❓❓❓
    ## 将变量放于对应数组中便于统一调用
S₁ = ComplexVariable(3,3)
    S₂ = ComplexVariable(3,3)
    S₃ = ComplexVariable(3,3)
    S₄ = ComplexVariable(3,3)
    S₅ = ComplexVariable(3,3)
    S₆ = ComplexVariable(3,3)
    S₇ = ComplexVariable(3,3)
    S₈ = ComplexVariable(3,3)
    S₉ = ComplexVariable(3,3)
    S₁₀= ComplexVariable(3,3)
    S₁₁= ComplexVariable(3,3)
S = vcat(S₁,S₂,S₃,S₄,S₅,S₆,S₇,S₈,S₉,S₁₀,S₁₁) # 支路复功率
s₁ = ComplexVariable(3)
    s₂ = ComplexVariable(3)
    s₃ = ComplexVariable(3)
    s₄ = ComplexVariable(3)
    s₅ = ComplexVariable(3)
    s₆ = ComplexVariable(3)
    s₇ = ComplexVariable(3)
    s₈ = ComplexVariable(3)
    s₉ = ComplexVariable(3)
    s₁₀= ComplexVariable(3)
    s₁₁= ComplexVariable(3)
    s₁₂= ComplexVariable(3)
s = vcat(s₁,s₂,s₃,s₄,s₅,s₆,s₇,s₈,s₉,s₁₀,s₁₁,s₁₂) # 电源发出功率sᵢ
l₁ = HermitianSemidefinite(3)
    l₂ = HermitianSemidefinite(3)
    l₃ = HermitianSemidefinite(3)
    l₄ = HermitianSemidefinite(3)
    l₅ = HermitianSemidefinite(3)
    l₆ = HermitianSemidefinite(3)
    l₇ = HermitianSemidefinite(3)
    l₈ = HermitianSemidefinite(3)
    l₉ = HermitianSemidefinite(3)
    l₁₀= HermitianSemidefinite(3)
    l₁₁= HermitianSemidefinite(3)
l = vcat(l₁,l₂,l₃,l₄,l₅,l₆,l₇,l₈,l₉,l₁₀,l₁₁) # 支路电流平方
v₁ = HermitianSemidefinite(3)
    v₂ = HermitianSemidefinite(3)
    v₄ = HermitianSemidefinite(3)
    v₃ = HermitianSemidefinite(3)
    v₅ = HermitianSemidefinite(3)
    v₆ = HermitianSemidefinite(3)
    v₇ = HermitianSemidefinite(3)
    v₈ = HermitianSemidefinite(3)
    v₉ = HermitianSemidefinite(3)
    v₁₀= HermitianSemidefinite(3)
    v₁₂= HermitianSemidefinite(3)
    v₁₁= HermitianSemidefinite(3)
v = vcat(v₁,v₂,v₃,v₄,v₅,v₆,v₇,v₈,v₉,l₁₀,v₁₁,v₁₂) # 节点电压平方
s₁_c = ComplexVariable(3)
    s₂_c = ComplexVariable(3)
    s₃_c = ComplexVariable(3)
    s₄_c = ComplexVariable(3)
    s₅_c = ComplexVariable(3)
    s₆_c = ComplexVariable(3)
    s₇_c = ComplexVariable(3)
    s₈_c = ComplexVariable(3)
    s₉_c = ComplexVariable(3)
    s₁₀_c= ComplexVariable(3)
    s₁₁_c= ComplexVariable(3)
    s₁₂_c= ComplexVariable(3)
s_c = vcat(s₁_c,s₂_c,s₃_c,s₄_c,s₅_c,s₆_c,s₇_c,s₈_c,s₉_c,s₁₀_c,s₁₁_c,s₁₂_c) # 电容器发出的功率

    # 确定目标函数，最小化网损，有功无功损耗都要考虑吧❓  #####只考虑有功🙂
#OPF_L = minimize(sum(real(Z[:,:,i]*l[i]) for i in 1:m))
OPF_L = minimize(sum(real(l₁*Z[:,:,1]+l₂*Z[:,:,2]+l₃*Z[:,:,3]+l₄*Z[:,:,4]+l₅*Z[:,:,5]+l₆*Z[:,:,6]+l₇*Z[:,:,7]+l₈*Z[:,:,8]+l₉*Z[:,:,9]+l₁₀*Z[:,:,10]+l₁₁*Z[:,:,11])))
    # 确定约束条件

    ######👇❌❌🤦❌❌👇考虑到s_c[100，100]也是存在的，我迷惑了，可能不能这样嵌套吧🙂
    #################开始胡写。###############

    # 分布式电源
    OPF_L.constraints += [real(sum(s[i])) <= real(sᵢₘ[i]/Sbase) for i=1:n] # 上限,P
    OPF_L.constraints += [0 <= real(sum(s[i])) for i=1:n] # 下限,P
    OPF_L.constraints += [imag(sum(s[i])) <= imag(sᵢₘ[i]) for i=1:n] # 上限,Q
    OPF_L.constraints += [0 <= imag(sum(s[i])) for i=1:n] # 下限,Q
    # 并联电容
    OPF_L.constraints += [real(s_c[i,1]) <= real(sᵢ𝒸[i,1]) for i=1:n] # 上限,A
    OPF_L.constraints += [0 <= real(s_c[i,1]) for i=1:n] # 下限,A
    OPF_L.constraints += [imag(s_c[i,2]) <= imag(sᵢ𝒸[i,2]) for i=1:n] # 上限,B
    OPF_L.constraints += [0 <= imag(s_c[i,2]) for i=1:n] # 下限,B
    OPF_L.constraints += [imag(s_c[i,3]) <= imag(sᵢ𝒸[i,3]) for i=1:n] # 上限,C
    OPF_L.constraints += [0 <= imag(s_c[i,3]) for i=1:n] # 下限,C
    # sⱼ[i]==s[i]+s_c[i]-sₗ[i] sⱼ(节点注入功率)=sᵢ(电源注入功率)+s_c(电容器发出功率)-sₗ(负载消耗功率)
    # 线路功率约束
    OPF_L.constraints += [
    v₁==v₁-(S₁*Z[:,:,1]'+Z[:,:,1]*S₁')+Z[:,:,1]*l₁*Z[:,:,1]'
    v₈==v₂-(S₂*Z[:,:,2]'+Z[:,:,2]*S₂')+Z[:,:,2]*l₂*Z[:,:,2]'
    v₃==v₂-(S₃*Z[:,:,3]'+Z[:,:,3]*S₃')+Z[:,:,3]*l₃*Z[:,:,3]'
    v₅==v₂-(S₄*Z[:,:,4]'+Z[:,:,4]*S₄')+Z[:,:,4]*l₄*Z[:,:,4]'
    v₄==v₃-(S₅*Z[:,:,5]'+Z[:,:,5]*S₅')+Z[:,:,5]*l₅*Z[:,:,5]'
    v₁₀==v₅-(S₆*Z[:,:,6]'+Z[:,:,6]*S₆')+Z[:,:,6]*l₆*Z[:,:,6]'
    v₆==v₅-(S₇*Z[:,:,7]'+Z[:,:,7]*S₇')+Z[:,:,7]*l₇*Z[:,:,7]'
    v₇==v₅-(S₈*Z[:,:,8]'+Z[:,:,8]*S₈')+Z[:,:,8]*l₈*Z[:,:,8]'
    v₉==v₈-(S₉*Z[:,:,9]'+Z[:,:,9]*S₉')+Z[:,:,9]*l₉*Z[:,:,9]'
    v₁₁==v₁₀-(S₁₀*Z[:,:,10]'+Z[:,:,10]*S₁₀')+Z[:,:,10]*l₁₀*Z[:,:,10]'
    v₁₀==v₁₀-(S₁₁*Z[:,:,11]'+Z[:,:,11]*S₁₁')+Z[:,:,11]*l₁₁*Z[:,:,11]']
    # 二阶锥约束
    OPF_L.constraints += [[v[i] S[i];S'[i] l[i]] in :SDP for i=1:n]
    #OPF_L.constraints += []
    #OPF_L.constraints += []

    ##❌❌❌❌运行后无解（必然的嗯）🤷‍❌❌❌❌
    solve!(OPF_L, () -> Mosek.Optimizer)
    println(evaluate(OPF_L))
