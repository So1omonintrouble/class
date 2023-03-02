#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
x = torch.rand(1,3)#定义矩阵1
y = torch.rand(2,1)#定义矩阵2
print(x)
print(y)
c = x-y;
print(x-y)#直接输出相减结果
print(torch.sub(x, y))#第二种减法形式
#第三种减法形式
print(c)


# In[ ]:




