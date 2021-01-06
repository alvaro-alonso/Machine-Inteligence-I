#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

n_classes = 10
batch_size = 100

train_set, test_set = [
    datasets.MNIST(
        './',
        train=train_flag,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ]),
    ) 
    for train_flag in [True, False]
]

train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test = torch.utils.data.DataLoader(test_set, batch_size=test_set.__len__(), shuffle=True)

train_example_data, train_example_targets = next(iter(train))
test_example_data, test_example_targets = next(iter(test))

print(f"train batch size: {len(train)} x {train_example_data.shape}")
print(f"test batch size: {len(test)} x {test_example_data.shape}")


# # Excercise 1

# In[2]:


import numpy as np
import torch.nn as nn
import torch.optim as optim

class Linear_Model(nn.Module):
    def __init__(self):
        super(Linear_Model, self).__init__()
        self.linear = nn.Linear(784, 10)
        self.linear.bias.data.fill_(0)
        
    def forward(self, x):
        h = self.linear(x)
        return F.softmax(h, dim=1) 


# In[3]:


learning_rate = 0.5
model = Linear_Model()    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# In[4]:



iterations = 10000
epochs = iterations // len(train)
test_data, test_targets = next(iter(test))

train_acc = []
test_acc = []

def accuracy(pred, target):
    winners = pred.argmax(dim=1)
    corrects = (winners == target)
    return corrects.sum().float() / float(target.size(0) )

for epoch in range(epochs):
    for idx, (data, target) in enumerate(train):
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        if idx % 100 == 0:
            train_accuracy = accuracy(output, target)
            train_acc.append(train_accuracy.item())
            
            val_output = model(test_data)
            test_accuracy = accuracy(val_output, test_targets)
            test_acc.append(test_accuracy.item())
        
        
    print('epoch {}, loss {}'.format(epoch, loss.item()))


# In[5]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

epochs = range(len(train_acc))
plt.plot(epochs, train_acc, label="Train Acc")
plt.plot(epochs, test_acc, label="Test Acc")
plt.legend()


# # Excercise 2

# In[26]:


# function replicated from: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/20
# not working quite ... need improvement
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def weights_init(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0.1)
        truncated_normal_(m.weight, 0, 0.01)
        
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 1500)
        self.linear2 = nn.Linear(1500, 1500)
        self.linear3 = nn.Linear(1500, 1500)
        self.linear4 = nn.Linear(1500, 10)
        self.apply(weights_init)
        
    def forward(self, x):
        relu = nn.ReLU()
        h1 = relu(self.linear1(x))
        h2 = relu(self.linear2(h1))
        h3 = relu(self.linear3(h2))
        # the function softmax is not included as nn.CrossEntropyLoss applies a softmax (source: https://discuss.pytorch.org/t/pytorch-equivalence-to-sparse-softmax-cross-entropy-with-logits-in-tensorflow/18727/2)
        return self.linear4(h3) 
    


# In[28]:


learning_rate, betas, eps = 0.001, [0.9, 0.999], 1e-8

model = MLP()    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas)


# In[29]:


iterations = 20000
epochs = iterations // len(train)
test_data, test_targets = next(iter(test))

train_acc = []
test_acc = []

def accuracy(pred, target):
    winners = pred.argmax(dim=1)
    corrects = (winners == target)
    return corrects.sum().float() / float(target.size(0) )

for epoch in range(epochs):
    for idx, (data, target) in enumerate(train):
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        if idx % 100 == 0:
            train_accuracy = accuracy(output, target)
            train_acc.append(train_accuracy.item())
            
            val_output = model(test_data)
            test_accuracy = accuracy(val_output, test_targets)
            test_acc.append(test_accuracy.item())
        
        
    print('epoch {}, loss {}'.format(epoch, loss.item()))


# In[30]:


epochs = range(len(train_acc))
plt.plot(epochs, train_acc, label="Train Acc")
plt.plot(epochs, test_acc, label="Test Acc")
plt.legend()


# # Excercise 3

# In[31]:


class MLP_w_DOUT(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 1500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1500, 1500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1500, 1500),
            nn.ReLU(),
            nn.Dropout(),
            # the function softmax is not included as nn.CrossEntropyLoss applies a softmax (source: https://discuss.pytorch.org/t/pytorch-equivalence-to-sparse-softmax-cross-entropy-with-logits-in-tensorflow/18727/2)
            nn.Linear(1500, 10)
        )
        self.apply(weights_init)
        
    def forward(self, x):
        return self.network(x) 


# In[ ]:




