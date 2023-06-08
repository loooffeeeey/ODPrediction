from atexit import register
import copy
from random import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset, random_split

import dgl


def data_split(full_dataset, config):
    num_all = len(full_dataset)
    train_size = int(num_all * config["train"])
    valid_size = int(num_all * config["valid"])
    test_size = num_all - train_size - valid_size
    train, valid, test = random_split(full_dataset, [train_size, valid_size, test_size])
    return train, valid, test

def build_DGLGraph(adj):
    edges = adj.nonzero()
    g = dgl.graph(edges)
    return g


class Urban_dataset(Dataset):
    def __init__(self, config):
        super(Urban_dataset, self).__init__()

        self.config = config

    def getData(self, data):
        # numpy
        self.region_attributes = data["region_attr"].drop(columns=["geometry", "adcode", "area", "extensibil", "compactnes"]).values.astype(np.float32)
        self.distance = data["distance"].astype(np.float32)
        self.adjacency = self.dis_2_adj(self.distance).astype(np.float32)
        self.OD = data["OD"].astype(np.float32)
        self.traffic_graph = data["traffic_graph"].astype(np.float32)
        self.speeds = data["speed"].astype(np.float32)

        self.data_minmax()
        
        if self.config["baseline"] == "gravity":
            pass
        else:
            self.node_feats = self.region_attr_con_TIME(self.region_attributes)
    
    def data_minmax(self):
        # distance
        self.distance_raw = copy.deepcopy(self.distance)
        self.distance = self.distance.reshape([-1, 1])
        dis_scaler = MinMaxScaler(feature_range=(1, 2))
        dis_scaler.fit(self.distance)
        self.distance = dis_scaler.transform(self.distance).reshape([self.region_attributes.shape[0], self.region_attributes.shape[0]])
        self.dis_scaler = dis_scaler
        # region_attributes
        self.region_attributes_raw = self.region_attributes
        region_scaler = MinMaxScaler(feature_range=(-1, 1))
        region_scaler.fit(self.region_attributes)
        self.region_attributes = region_scaler.transform(self.region_attributes)
        self.region_scaler = region_scaler
        # OD
        nonzero_OD = self.OD[self.OD.nonzero()]
        OD_max, OD_min = nonzero_OD.max(), nonzero_OD.min()
        self.OD_minmax = OD_min, OD_max
        self.OD_raw = copy.deepcopy(self.OD)
        self.OD = (self.OD - OD_min) *2 / (OD_max - OD_min) - 1

    def region_attr_con_TIME(self, region_attr):
        node_feats = []
        for i in range(24):
            onehot = np.zeros([region_attr.shape[0], 24])
            onehot[:, i] = 1
            tmp_node_feats = np.concatenate((copy.deepcopy(region_attr), onehot), axis=1)
            node_feats.append(tmp_node_feats)
        node_feats = np.stack(node_feats)
        return node_feats

    def dis_2_adj(self, dis):
        adj = copy.deepcopy(dis)
        adj[adj > 100] = 0
        adj[adj != 0] = 1
        return adj
        

class IDX_datasets(Dataset):
    def __init__(self, config):
        super(IDX_datasets, self).__init__()
        self.config = config

    def __getitem__(self, i):
        x_idx = self.OD_pair_index[:, i]
        y = self.OD[x_idx[0], x_idx[1], :]
        return x_idx, y

    def __len__(self):
        return self.OD_pair_index.shape[1]

    def getOD(self, OD):
        self.OD = OD
        self.OD_pair_index = self.get_OD_pair_index()
        return OD

    def get_OD_pair_index(self):
        return np.array(self.OD.mean(2).nonzero())


    