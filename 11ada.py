#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:56:42 2018

@author: pan
"""

import numpy as np
import pandas as pd

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

data=pd.read_csv("./data/datatrain.csv",encoding = "GBK")
#分割测试数据和训练数据
test, train = train_test_split(data, train_size=0.3, random_state=1)

#训练数据
y=train['tag']
X=train
del train['tag']

#测试数据
TagTest=test['tag']
XdataTest=test
del XdataTest['tag']


# 使用随机森林作为分类器，分类器有20课树
clf = AdaBoostRegressor()


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

base_estimator=[DecisionTreeRegressor(max_depth=x) for x in np.linspace(1,10,10)]
# 设置可能学习的参数
param_dist = {"base_estimator": base_estimator,
              "n_estimators": [int(i) for i in np.linspace(30,130,11)],
              "loss":["linear","square","exponential"]}

# 随机搜索， randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
#起始时间
start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)
print("=============================================")

# use a full grid over all parameters
param_grid = {"base_estimator": base_estimator,
              "n_estimators": [int(i) for i in np.linspace(30,130,11)],
              "loss":["linear","square","exponential"]}

# 网格搜索， grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

#对比分类效果
#random_search
#预测结果        
print("============================random_search=========================================")
Result = random_search.predict(XdataTest)
mse = np.average((Result - np.array(TagTest)) ** 2)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
print ("Mean Squared Error is:",mse)
print ("Root Mean Squared Error is:",rmse)

t = np.arange(len(XdataTest))
plt.figure()
plt.plot(t, TagTest, 'r-', linewidth=2, label='Test')
plt.plot(t, Result, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()

#预测结果        
print("============================grid_search=========================================")
Result = grid_search.predict(XdataTest)
mse = np.average((Result - np.array(TagTest)) ** 2)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
print ("Mean Squared Error is:",mse)
print ("Root Mean Squared Error is:",rmse)
plt.figure()
t = np.arange(len(XdataTest))
plt.plot(t, TagTest, 'r-', linewidth=2, label='Test')
plt.plot(t, Result, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()