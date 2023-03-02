#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import warnings
from torch.utils.data import DataLoader,Dataset
warnings.filterwarnings("ignore") 
import random  
from IPython import display  
from matplotlib import pyplot as plt    
from PIL import Image  
import os  
from torch import nn  
import torch.optim as optim  
from torch.nn import init  
import torch.nn.functional as F  
import time  
import pandas as pd  
from sklearn.utils import shuffle  
import math  
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae  
import datasets
import utils
from sklearn.metrics import mean_squared_error as mse_fn, mean_absolute_error as mae_fn


def mape_fn(y, pred):
    mask = y != 0
    y = y[mask]
    pred = pred[mask]
    mape = np.abs((y - pred) / y)
    mape = np.mean(mape) * 100
    return mape


def eval(y, pred):
    y = y.cpu().numpy()
    pred = pred.cpu().numpy()
    mse = mse_fn(y, pred)
    rmse = math.sqrt(mse)
    mae = mae_fn(y, pred)
    mape = mape_fn(y, pred)
    return [rmse, mae, mape]

# 定义dataset
class my_Dataset(data.Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.y = labels

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]
    
class TrafficDataset:
    def __init__(self):
        self.raw_data = np.load(r'C:\Users\20693\Desktop\实验4_22121501_束天成\dataset\Traffic\Traffic.npz')['data']
        # self.raw_data = pd.DataFrame(self.raw_data)
        # 数据标准化
        self.min = self.raw_data.min()
        self.max = self.raw_data.max()
        self.data = (self.raw_data - self.min) / (self.max - self.min)

    def denormalize(self, x):
        return x * (self.max - self.min) + self.min

    def construct_set(self, train_por=0.6, test_por=0.2, window_size=12, label=0):
        train_x = []
        train_y = []
        val_x = []
        val_y = []
        test_x = []
        test_y = []
        window_size = 12
        len_train = int(self.data.shape[0] * 0.6)
        train_seqs = self.data[:len_train]
        for i in range(train_seqs.shape[0] - window_size):
            train_x.append(train_seqs[i:i + window_size].squeeze())
            train_y.append(train_seqs[i + window_size].squeeze())

        len_val = int(self.data.shape[0] * 0.8)
        val_seqs = self.data[len_train:len_val]
        for i in range(val_seqs.shape[0] - window_size):
            val_x.append(train_seqs[i:i + window_size].squeeze())
            val_y.append(train_seqs[i + window_size].squeeze())

        test_seqs = self.data[len_val:]
        for i in range(test_seqs.shape[0] - window_size):
            test_x.append(test_seqs[i:i + window_size].squeeze())
            test_y.append(test_seqs[i + window_size].squeeze())

        train_set = my_Dataset(torch.Tensor(train_x).unsqueeze(-1), torch.Tensor(train_y))
        val_set = my_Dataset(torch.Tensor(val_x).unsqueeze(-1), torch.Tensor(val_y))
        test_set = my_Dataset(torch.Tensor(test_x).unsqueeze(-1), torch.Tensor(test_y))
        return train_set, val_set, test_set
#补全相关的代码

batch_size = 64    
TrafficData = TrafficDataset()
train_set,val_set,test_set = TrafficData.construct_set(0.6,0.2,12)
train_loader = data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
val_loader = data.DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
test_loader = data.DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
print(train_loader)
print(val_loader)
print(test_loader)


# In[ ]:




