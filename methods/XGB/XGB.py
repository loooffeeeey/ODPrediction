import xgboost as xgb
import numpy as np
import pickle

import time
import setproctitle
setproctitle.setproctitle('xgb@rongcan')

od = np.load('/data/rongcan/data/ODGen_transfer/data/od/NYC/od_2015.npy')
dem = np.load('/data/rongcan/data/ODGen_transfer/data/regions/NYC/Demographics/2015/dem_nyc_2015_0.9.npy')
job = np.load('/data/rongcan/data/ODGen_transfer/data/regions/NYC/LODES/job_2015.npy')
poi = np.load('/data/rongcan/data/ODGen_transfer/data/regions/NYC/POI/poi.npy')
dis = np.load('/data/rongcan/data/ODGen_transfer/data/adj/NYC/distance.npy')
feat = np.concatenate((dem, job, poi), axis=1)

train_index = pickle.load(open('/data/rongcan/data/GMEL_AAAI2020/LODES/train_index.pkl', 'rb'))
test_index = pickle.load(open('/data/rongcan/data/GMEL_AAAI2020/LODES/test_index.pkl', 'rb'))

train_dis = dis[train_index].reshape([-1, 1])
train_x = np.concatenate( (feat[train_index[0]], feat[train_index[1]], train_dis), axis=1)
train_y = od[train_index]

test_dis = dis[test_index].reshape([-1, 1])
test_x = np.concatenate( (feat[test_index[0]], feat[test_index[1]], test_dis), axis=1)
test_y = od[test_index]

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

dtrain = xgb.DMatrix(train_x, label=train_y)
evallist = [(dtrain, 'train')]

param = {'max_depth': 8, 'eta': 0.5, 'objective': 'reg:squarederror'} # binary:hinge binary:logistic
param['eval_metric'] = 'rmse'

param['gpu_id'] = 0
param['tree_method'] = 'gpu_hist'
# param['alpha'] = 0.5
# param['lambda'] = 5
# param['gamma'] = 0.5
# param['subsample'] = 0.8
param['max_delta_step'] = 2
param['min_child_weight'] = 4

num_round = 400
bst = xgb.train(param, dtrain, num_round, evallist)

dtest = xgb.DMatrix(test_x)
prediction = bst.predict(dtest)
def RMSE(pre, gt):
    return np.sqrt(np.mean((pre - gt)**2))
print('RMSE = ', RMSE(prediction, test_y))
