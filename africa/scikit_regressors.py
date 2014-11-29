
# coding: utf-8
# @author: Maria Rigaki
# Testing various regressors from the scikit-learn package

import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcess

from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model

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

print np.shape(xtrain)
#est = svm.SVR(C=700.0, verbose=0, epsilon=0.01)
est = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=2, max_depth=6)
#est = AdaBoostRegressor(n_estimators=100, learning_rate=0.1,
#                        loss='square', random_state=1234)

#est = linear_model.BayesianRidge(alpha_1=1e-05, alpha_2=1e-05,
#                                 lambda_1=1e-05, lambda_2=1e-05, verbose=1)
#est = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
#                                max_depth=10, random_state=0, loss='ls', verbose=1)

#est = linear_model.LogisticRegression(C=1.1)
preds = np.zeros((xtest.shape[0], 5))
scores = np.zeros(5)

#for i in range(5):
    #sup_vec.fit(xtrain, labels[:, i])
#    est.fit(xtrain, labels[:, i])
#    xtrain_new = est.transform(xtrain)
#    xtest_new = est.transform(xtest)

kf = cross_validation.KFold(len(xtrain), n_folds=5)
for i in range(5):
    score = 0
    for train_index, test_index in kf:
        est.fit(xtrain[train_index], labels[train_index, i])
        score += np.sqrt(mse(labels[test_index, i], est.predict(xtrain[test_index]).astype(float)))
    print i, score/5.0
    scores[i] = score/5.0

# for i in range(5):
#     est.fit(xtrain, labels[:, i])
#     preds[:, i] = est.predict(xtest).astype(float)
#
# sample = pd.read_csv('data/sample_submission.csv')
# sample['Ca'] = preds[:, 0]
# sample['P'] = preds[:, 1]
# sample['pH'] = preds[:, 2]
# sample['SOC'] = preds[:, 3]
# sample['Sand'] = preds[:, 4]
#
# sample.to_csv('result_SVM.csv', index=False)
print "k-fold CV score: ", np.mean(scores)


