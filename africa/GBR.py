
# coding: utf-8
# @author: Maria Rigaki
# Gradient Boosting Regressor pararameter search using GridCV

import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('data/training.csv')
test = pd.read_csv('data/sorted_test.csv')
labels = train[['Ca', 'P', 'pH', 'SOC', 'Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)
co2_bands = ['m2379.76', 'm2377.83', 'm2375.9', 'm2373.97', 'm2372.04', 'm2370.11',
             'm2368.18', 'm2366.26', 'm2364.33', 'm2362.4',  'm2360.47', 'm2358.54',
             'm2356.61', 'm2354.68', 'm2352.76']
train.drop(co2_bands, axis=1, inplace=True)
test.drop(co2_bands, axis=1, inplace=True)

xtrain, xtest = np.array(train)[:, :3578], np.array(test)[:, :3578]


#sup_vec = svm.SVR(C=10000.0, verbose=2)
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=5, random_state=0, loss='ls', verbose=0)

preds = np.zeros((xtest.shape[0], 5))
scores = np.zeros(5)

grid_params = {'max_depth': [5],
               'learning_rate': [0.001, 0.01],
               'n_estimators': [100]}
grid = GridSearchCV(est, param_grid=grid_params, cv=5, verbose=1, n_jobs=2, scoring='mean_squared_error')
for i in range(5):
    grid.fit(xtrain, labels[:, i])
    print i, grid.best_params_, grid.best_score_, np.sqrt(-grid.best_score_)
    scores[i] = np.sqrt(-grid.best_score_)
    preds[:, i] = grid.predict(xtest).astype(float)

# for i in range(5):
#     #sup_vec.fit(xtrain, labels[:, i])
#     est.fit(xtrain, labels[:, i])
#     preds[:, i] = est.predict(xtest).astype(float)
#
sample = pd.read_csv('data/sample_submission.csv')
sample['Ca'] = preds[:, 0]
sample['P'] = preds[:, 1]
sample['pH'] = preds[:, 2]
sample['SOC'] = preds[:, 3]
sample['Sand'] = preds[:, 4]

sample.to_csv('result_Grid.csv', index=False)
print np.mean(scores)


