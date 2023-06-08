import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from dgl.nn import GraphConv, GATConv

from tqdm import tqdm

from utils.metrics import *

class GNN(nn.Module):
    def __init__(self, config, out_dim):
        super(GNN, self).__init__()
        self.conv_in = GraphConv(config["GNN_in_dim"], config["GNN_hid_dim"])
        self.conv_hid = GraphConv(config["GNN_hid_dim"], config["GNN_hid_dim"])
        self.conv_out = GraphConv(config["GNN_hid_dim"], out_dim)

    def forward(self, g, x):
        h = torch.relu(self.conv_in(g, x))
        h = torch.relu(self.conv_hid(g, h))
        out = self.conv_out(g, h)
        return out

class one_gnn(nn.Module):
    def __init__(self, config):
        super(one_gnn, self).__init__()

        self.gnn = GNN(config, config["node_embsize"])
        self.linear_out = nn.Linear(config["node_embsize"]*2+1, 1)

    def forward(self, g, x, dis, train_idx):
        g_emb = torch.sigmoid(self.gnn(g, x))
        o_emb = g_emb[train_idx[0]]
        d_emb = g_emb[train_idx[1]]
        dis_emb = dis[train_idx].reshape([-1, 1])
        emb = torch.cat((o_emb, d_emb, dis_emb), dim=1)
        pre = torch.tanh(self.linear_out(emb))
        return pre

class gnn(nn.Module):
    def __init__(self, config):
        super(one_gnn, self).__init__()

        self.gnn = gnn(config)

    def forward(self, g, x , dis, train_idx):
        pres = []
        for i in range(24):
            pre = self.gnn(g, x[i], dis, train_idx).squeeze()
            pres.append(pre)
        pres = torch.stack(pres, dim=1)
        return pres


def GNN(config, input, trainDataloader, validDataloader, testDataloader):
    print("----- one-gnn -----")
    # model
    model = gnn(config)
    model.to(config["device"])

    # train
    criterion = nn.MSELoss()
    optm = Adam(list(model.parameters()), lr=3e-4)
    for epoch in tqdm(range(20000)):
        loss_epoch = []
        for od_pair_idx, od_flow in trainDataloader:
            optm.zero_grad()
            od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
            od_flow = od_flow.float().to(config["device"])
            pre = model(input["region_graph"],
                        input["region_node_feats"],
                        input["distance"],
                        od_pair_idx)
            loss = criterion(od_flow, pre)
            loss_epoch.append(loss.item())
            loss.backward()
            optm.step()
        loss_epoch = float(np.sqrt(np.mean(loss_epoch)))

    # test
    with torch.no_grad():
        od_pair_idx, od_flow = next(iter(testDataloader))
        od_pair_idx = (od_pair_idx[:,0], od_pair_idx[:,1])
        od_flow = od_flow.float().to(config["device"])
        pre = model(input["region_graph"],
                    input["region_node_feats"],
                    input["distance"],
                    od_pair_idx)
        pre = reMinMax(pre, input["OD_minmax"])
        od_flow = reMinMax(od_flow, input["OD_minmax"])
        rmse = float(RMSE(pre, od_flow))
        nrmse = float(NRMSE(pre, od_flow))
        mae = float(MAE(pre, od_flow))
        mape = float(MAPE(pre, od_flow))
        smape = float(SMAPE(pre, od_flow))
        cpc = float(CPC(pre, od_flow))
        print("RMSE = ", rmse)
        print("NRMSE = ", nrmse)
        print("mae = ", mae)
        print("mape = ", mape)
        print("smape = ", smape)
        print("cpc=", cpc)