import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import torch
import time
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import setproctitle
setproctitle.setproctitle('MLP@rongcan')

device = torch.device('cuda:{}'.format(1))

od = np.load('/data/rongcan/data/ODGen_transfer/data/od/NYC/od_2015.npy')
dem = np.load('/data/rongcan/data/ODGen_transfer/data/regions/NYC/Demographics/2015/dem_nyc_2015_0.9.npy')
job = np.load('/data/rongcan/data/ODGen_transfer/data/regions/NYC/LODES/job_2015.npy')
poi = np.load('/data/rongcan/data/ODGen_transfer/data/regions/NYC/POI/poi.npy')
dis = np.load('/data/rongcan/data/ODGen_transfer/data/adj/NYC/distance.npy')
feat = np.concatenate((dem, job, poi), axis=1)

feats_max = np.max(feat, axis=0)
feats_min = np.min(feat, axis=0)
feat = (feat - feats_min)*2 / (feats_max-feats_min) - 1

train_index = pickle.load(open('/data/rongcan/data/GMEL_AAAI2020/LODES/train_index.pkl', 'rb'))
test_index = pickle.load(open('/data/rongcan/data/GMEL_AAAI2020/LODES/test_index.pkl', 'rb'))

train_dis = dis[train_index].reshape([-1, 1])
train_x = np.concatenate( (feat[train_index[0]], feat[train_index[1]], train_dis), axis=1)
train_y = od[train_index]

test_dis = dis[test_index].reshape([-1, 1])
test_x = np.concatenate( (feat[test_index[0]], feat[test_index[1]], test_dis), axis=1)
test_y = od[test_index]

# od_max = np.max(od)
# od_min = np.min(od)
# train_y = (train_y - od_min)*2 / (od_max - od_min) -1

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
print(np.sum(train_x), np.max(train_x), np.min(train_x))
print(np.sum(train_y), np.max(train_y), np.min(train_y))
print(np.sum(test_x), np.max(test_x), np.min(test_x))
print(np.sum(test_y), np.max(test_y), np.min(test_y))
# exit(0)

class MLP(nn.Module):
    def __init__(self, in_dim, hid, out):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid)
        self.linear2 = nn.Linear(hid, out)

        self.bn1 = nn.BatchNorm1d(hid)
        self.bn2 = nn.BatchNorm1d(out)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        # x = self.bn1(x)
        x = torch.relu(self.linear2(x))
        # x = self.bn2(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.MLP1 = MLP(174, 256, 8)
        self.MLP2 = MLP(174, 256, 8)
        self.MLP3 = MLP(1, 8, 1)
        self.linear = nn.Linear(8*2 + 1, 1)

    def forward(self, x):
        x1 = x[:,:int(x.size(1)/2)]
        x2 = x[:,int(x.size(1)/2):-1]
        d = x[:,-1].view([-1, 1])

        x1 = self.MLP1(x1)
        x2 = self.MLP2(x2)
        x3 = self.MLP3(d)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.linear(x)
        # x = torch.relu(self.linear(x))
        return x

train_x = torch.FloatTensor(train_x).to(device)
train_y = torch.FloatTensor(train_y).to(device)
test_x = torch.FloatTensor(test_x).to(device)
# test_y = torch.FloatTensor(test_y)

dataset = TensorDataset(train_x, train_y)
Loader = DataLoader(dataset = dataset,
                               batch_size = 10000,
                               shuffle = True)

print('Start training...')
start = time.time()
#train
epoch = 50
model = Model().to(device)
model.train()
loss_list = []
optimizer = optim.Adam(model.parameters(), lr=0.00005)
criterion = nn.MSELoss().cuda()
for i in range(epoch):
    for i, x_y in enumerate(Loader):
        x, y = x_y
        optimizer.zero_grad()
        pre = model(x)
        loss = criterion(pre, y)

        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        print('loss = ', loss.item())

def RMSE(pre, gt):
    return np.sqrt(np.mean((pre - gt)**2))

prediction = model(test_x).cpu().detach().numpy().reshape([-1])
# prediction = (prediction + 1) *(od_max - od_min) / 2 + od_min
print('Consume ', time.time()-start, 'seconds.', 'RMSE = ', RMSE(prediction, test_y))
