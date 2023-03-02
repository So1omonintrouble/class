#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import time
#导入做实验必要的数据包等相关内容

batch_size = 128
num_classes = 3
lr = 0.001
epochs = 10 
device = torch.device("cpu")

file_root = "D:\实验3数据包\数据包\车辆分类数据集"
classes = ['bus', 'car', 'truck']  # 分别对应三个标签
nums = [218, 779, 360]  # 每种类别的个数

def read_data(path):
    file_name = os.listdir(path)  # 打开对应的文件夹
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    train_num = [int(num * 4 / 5) for num in nums]#定义训练集数量
    test_num = [nums[i] - train_num[i] for i in range(len(nums))]#定义测试集数量
    
    for idx, f_name in enumerate(file_name):  # 定义数据标签等相关内容
        im_dirs = path + '/' + f_name
        im_path = os.listdir(im_dirs)  # 每个不同类别图像文件夹下所有图像的名称

        index = list(range(len(im_path)))
        random.shuffle(index)  # 打乱顺序
        im_path_ = list(np.array(im_path)[index])
        test_path = im_path_[:test_num[idx]]  # 测试数据的路径
        train_path = im_path_[test_num[idx]:]  # 训练数据的路径
        
        for img_name in train_path:
            if img_name == 'desktop.ini':
                continue
            img = Image.open(im_dirs + '/' + img_name)  # 打开图片
            img = img.resize((64, 64), Image.ANTIALIAS)  # 对图片进行变形
            train_data.append(img)
            train_labels.append(idx)

        for img_name in test_path:
            if img_name == 'desktop.ini':
                continue
            img = Image.open(im_dirs + '/' + img_name)  # 打开图片
            img = img.resize((64, 64), Image.ANTIALIAS)  # 对图片进行变形
            test_data.append(img)
            test_labels.append(idx)

    print('训练集大小：', len(train_data), ' 测试集大小：', len(test_data))
    return train_data, train_labels, test_data, test_labels
train_data, train_labels, test_data, test_labels = read_data(file_root)

# 定义一个Transform操作
transform = transforms.Compose(
    [transforms.ToTensor(),  # 变为tensor
     # 对数据按通道进行标准化，即减去均值，再除以方差, [0-1]->[-1,1]
     transforms.Normalize(mean=[0.4686, 0.4853, 0.5193], std=[0.1720, 0.1863, 0.2175])
     ]
)#将数据类型进行转换，最终要转换成Tensor


# 自定义Dataset类实现每次取出图片，将PIL转换为Tensor
class MyDataset(Dataset):
    def __init__(self, data, label, trans):
        self.len = len(data)
        self.data = data
        self.label = label
        self.trans = trans
    def __getitem__(self, index):  # 根据索引返回数据和对应的标签
        return self.trans(self.data[index]), self.label[index]
    def __len__(self):
        return self.len
# 调用自己创建的Dataset
train_dataset = MyDataset(train_data, train_labels, transform)
test_dataset = MyDataset(test_data, test_labels, transform)

# 生成data loader
train_iter = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)#得到数据训练集
test_iter = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)#得到数据测试集

# 使用torch.nn实现二维卷积


#残差网络块
#每个残差块都是两层
#默认3*3卷积下padding为1，则大小不会变化，如变化则是步长引起的。

class ResidualBlock(nn.Module):
    def __init__(self, nin, nout, size, stride=1, shortcut=True):
        super(ResidualBlock, self).__init__()
        #两层卷积层
        #不同步长只有第一层卷积层不同
        self.block1 = nn.Sequential(nn.Conv2d(nin, nout, size, stride, padding=1),
                                    nn.BatchNorm2d(nout),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(nout, nout, size, 1, padding=1),
                                    nn.BatchNorm2d(nout))
        self.shortcut = shortcut
        #解决通道数变化以及步长不为1引起的图片大小的变化
        self.block2 = nn.Sequential(nn.Conv2d(nin, nout, size, stride, 1),
                                    nn.BatchNorm2d(nout))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        x = input
        out = self.block1(x)
        '''若输入输出维度相等直接相加，不相等改变输入的维度--包括大小和通道'''
        if self.shortcut:
            out = x + out
        else:
            out = out + self.block2(x)
        out = self.relu(out)
        return out


# In[ ]:


#定义给定的残差结构
class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        #t表示2个相同的残差块,每个残差块两个卷积
        self.d1 = self.make_layer(64, 64, 3, stride=1, t=2)
        self.d2 = self.make_layer(64, 128, 3, stride=2, t=2)
        self.d3 = self.make_layer(128, 256, 3, stride=2, t=2)
        self.d4 = self.make_layer(256, 512, 3, stride=2, t=2)

        self.avgp = nn.AvgPool2d(8)
        self.exit = nn.Linear(512, 3)

    def make_layer(self, in1, out1, ksize, stride, t):
        layers = []
        for i in range(0, t):
            if i == 0 and in1 != out1:
                layers.append(ResidualBlock(in1, out1, ksize, stride, None))
            else:
                layers.append(ResidualBlock(out1, out1, ksize, 1, True))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.block(input)  # 输出维度 64 * 64 * 64    C * H * W
        x = self.d1(x)  # 输出维度 64 * 54 * 54
        x = self.d2(x)  # i=0 步长为2，输出维度128 * 32 * 32
        x = self.d3(x)  # i=0 步长为2，输出维度256 * 16 * 16 
        x = self.d4(x)  # i=0 步长为2，输出维度512 * 8 * 8
        x = self.avgp(x)  # 512 * 1 * 1
      #将张量out从shape batchx512x1x1 变为 batch x512
        x = x.squeeze()
        output = self.exit(x)
        return output

#初始化模型
net = resnet().to(device)
#使用多元交叉熵损失函数
criterion = nn.CrossEntropyLoss()
loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
#使用Adam优化器
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

def train(net, data_loader, device):#模型训练
    net.train()  # 指定为训练模式
    train_batch_num = len(data_loader)
    total_loss = 0.0
    correct = 0  # 记录共有多少个样本被正确分类
    sample_num = 0

    # 遍历每个batch进行训练
    for data, target in data_loader:
        # 将图片和标签放入指定的device中
        data = data.to(device)
        target = target.to(device)
        # 将当前梯度清零
        optimizer.zero_grad()
        # 使用模型计算出结果
        y_hat = net(data)
        # 计算损失
        loss_ = loss(y_hat, target)
        # 进行反向传播
        loss_.backward()
        optimizer.step()
        total_loss += loss_.item()
        cor = (torch.argmax(y_hat, 1) == target).sum().item()
        correct += cor
        # 累加当前的样本总数
        sample_num += target.shape[0]
        print('loss: %.4f  acc: %.4f' % (loss_.item(), cor/target.shape[0]))
    # 平均loss和准确率
    loss_ = total_loss / train_batch_num
    acc = correct / sample_num
    return loss_, acc

# 对模型进行测试
def test(net, data_loader, device):
    net.eval()  # 指定当前模式为测试模式（针对BN层和dropout层）
    test_batch_num = len(data_loader)
    total_loss = 0
    correct = 0
    sample_num = 0
    # 指定不进行梯度计算（没有反向传播也会计算梯度，增大GPU开销
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            loss_ = loss(output, target)
            total_loss += loss_.item()
            correct += (torch.argmax(output, 1) == target).sum().item()
            sample_num += target.shape[0]
    loss_ = total_loss / test_batch_num
    acc = correct / sample_num
    return loss_, acc

# 模型训练与测试
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
time_list = []  
timestart = time.clock()

for epoch in range(num_epochs):
    epochstart = time.clock()
    # 在训练集上训练
    train_loss, train_acc = train(net, data_loader=train_iter, device=device)
    # 测试集上验证
    test_loss, test_acc = test(net, data_loader=test_iter, device=device)
    elapsed = (time.clock() - epochstart)
    
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    time_list.append(elapsed) 
    print('epoch %d, train loss: %.4f, train acc: %.3f' % (epoch+1, train_loss, train_acc))
    print('test loss: %.4f, test acc: %.3f' % (test_loss, test_acc))
    print('Time used %.4f' % (elapsed))
    
timesum1 = (time.clock() - timestart)  
print('Time used %.4f' % (timesum1))
    
    
# 绘制函数
def draw_(x, train_Y, test_Y, ylabel):
    plt.plot(x, train_Y, label='train_' + ylabel, linewidth=1.5)
    plt.plot(x, test_Y, label='test_' + ylabel, linewidth=1.5)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.legend()  # 加上图例
    plt.show()
# 绘制loss曲线
x = np.linspace(0, len(train_loss_list), len(train_loss_list))
draw_(x, train_loss_list, test_loss_list, 'loss')
draw_(x, train_acc_list, test_acc_list, 'accuracy')


# In[ ]:




