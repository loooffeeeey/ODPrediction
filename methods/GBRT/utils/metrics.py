import numpy as np

import torch


def reMinMax(data, MinMax):
    min_, max_ = MinMax
    return (data + 1) * (max_-min_) / 2 + min_

# for pytorch
def RMSE(x,y):
    return torch.sqrt(((x - y) **2).mean())

def NRMSE(x, y):
    return RMSE(x, y) / torch.std(y)

def MAE(x, y):
    return torch.abs(x - y).mean()

def MAPE(x, y):
    return (torch.abs(x - y) / (y + 1)).mean()

def SMAPE(x, y):
    return (torch.abs(x - y) / ((torch.abs(x) + torch.abs(y)) / 2 + 1)).mean()

def CPC(x, y):
    min_, _ = torch.cat((x.reshape([-1,1]), y.reshape([-1,1])), dim=1).min(1)
    return (2 * min_.sum()) / ( x.sum() +y.sum() )

# for numpy
def RMSE_np(x,y):
    return np.sqrt(((x - y) **2).mean())

def NRMSE_np(x, y):
    return RMSE_np(x, y) / np.std(y)

def MAE_np(x, y):
    return np.abs(x - y).mean()

def MAPE_np(x, y):
    return (np.abs(x - y) / (y + 1)).mean()

def SMAPE_np(x, y):
    return (np.abs(x - y) / ((np.abs(x) + np.abs(y)) / 2 + 1)).mean()

def CPC_np(x, y):
    min_ = np.concatenate((x.reshape([-1,1]), y.reshape([-1,1])), axis=1).min(1)
    return (2 * min_.sum()) / ( x.sum() +y.sum() )