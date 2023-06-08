import numpy as np
import time
import pickle

from sklearn.ensemble import RandomForestRegressor
# from sklearn.externals import joblib

import setproctitle
setproctitle.setproctitle('b-rf@rongcan')

city = 'LA'

od = np.load('/data/rongcan/data/ODGen_transfer/data/od/'+city+'/od_2015.npy')
dem = np.load('/data/rongcan/data/ODGen_transfer/data/regions/'+city+'/Demographics/2015/dem_2015_0.9.npy')
job = np.load('/data/rongcan/data/ODGen_transfer/data/regions/'+city+'/LODES/job_2015.npy')
poi = np.load('/data/rongcan/data/ODGen_transfer/data/regions/'+city+'/POI/poi.npy')
dis = np.load('/data/rongcan/data/ODGen_transfer/data/adj/'+city+'/distance.npy')
# pluto = np.load('/data/rongcan/data/GMEL_AAAI2020/PLUTO/PLUTO.npy')
feat = np.concatenate((dem, job, poi), axis=1)
print(feat.shape)
# feat = pluto

train_index = pickle.load(open('/data/rongcan/data/ODGen_transfer/data/od/'+city+'/train_index.pkl', 'rb'))
test_index = pickle.load(open('/data/rongcan/data/ODGen_transfer/data/od/'+city+'/test_index.pkl', 'rb'))

train_dis = dis[train_index].reshape([-1, 1])
train_x = np.concatenate( (feat[train_index[0]], feat[train_index[1]], train_dis), axis=1) # , train_dis
train_y = od[train_index]

test_dis = dis[test_index].reshape([-1, 1])
test_x = np.concatenate( (feat[test_index[0]], feat[test_index[1]], test_dis), axis=1) # , test_dis
test_y = od[test_index]

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

random_forest = RandomForestRegressor(n_estimators = 100,
                                      oob_score = True,
                                      max_depth = None,
                                      min_samples_split = 10,
                                      min_samples_leaf = 3,
                                      n_jobs = 48)

print('Start fitting...')
start = time.time()
random_forest.fit(X = train_x, y = train_y)
print('complete!')
print('Consume ', time.time()-start, ' seconds!')

def RMSE(pre, gt):
    return np.sqrt(np.mean((pre - gt)**2))

prediction = random_forest.predict(test_x)
np.save('prediction_rf.npy', prediction)
print('RMSE = ', RMSE(prediction, test_y))
