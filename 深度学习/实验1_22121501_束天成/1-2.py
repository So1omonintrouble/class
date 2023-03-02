#!/usr/bin/env python
# coding: utf-8

# In[14]:


import torch
P=torch.normal(0,0.01,size=(3,2))#创建第一个矩阵
Q=torch.normal(0,0.01,size=(4,2))#创建第二个矩阵
print(P)
print(Q)
QT=Q.t()#矩阵转置
print(QT)
S=P.mm(QT)#对新矩阵进行乘法运算
print(S)


# In[ ]:





# In[ ]:




