import numpy as np
import time
import setproctitle
import pickle

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.externals import joblib

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

GBRT = GradientBoostingRegressor(n_estimators = 3,
                                      max_depth = None,
                                      min_samples_split = 10,
                                      min_samples_leaf = 1)

print('Start fitting...')
start = time.time()
GBRT.fit(X = train_x, y = train_y)
print('complete!')
print('Consume ', time.time()-start, ' seconds!')

def RMSE(pre, gt):
    return np.sqrt(np.mean((pre - gt)**2))

prediction = GBRT.predict(test_x)
print('RMSE = ', RMSE(prediction, test_y))
