#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
x = torch.tensor(1.0, requires_grad=True)

y1 = x * x
y2 = x * x * x
y3 = y1 + y2

with torch.no_grad():
    y2 = x * x * x
    print(y3.requires_grad)
    print(y2.requires_grad)
    print(y1.requires_grad)
    y3.backward()
    print(x.grad)


# In[ ]:




