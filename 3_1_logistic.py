# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:40:38 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
import time
import datetime
import os

from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle,build_train_dataset,cal_log_loss
from smooth import BayesianSmoothing
import gen_smooth_features as smooth_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import minmax_scale
import xgboost as xgb
import operator 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    t0 = time.time()
    train_data = load_pickle(path=cache_pkl_path +'train_data_onehot')
    train_Y = load_pickle(path=cache_pkl_path +'train_Y')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data_onehot')
    cv_Y = load_pickle(path=cache_pkl_path +'cv_Y')
    
    test_data = load_pickle(path=cache_pkl_path +'test_data_onehot')
    test_file = 'round1_ijcai_18_test_a_20180301.txt'
    test = pd.read_table(raw_data_path + test_file,delim_whitespace=True)
    test_id = test.instance_id
    
    drop_cols = ['user_id','shop_id','item_id','item_brand_id','item_city_id']
    train_data.drop(drop_cols,axis=1,inplace=True)
    cv_data.drop(drop_cols,axis=1,inplace=True)
    test_data.drop(drop_cols,axis=1,inplace=True)
    
    
    print('train shap:',train_data.shape)
    print('cv shape', cv_data.shape)
    print('test shape', test_data.shape)
    
#    clf = LogisticRegression(C=1, fit_intercept=True, max_iter=10000,class_weight={0:0.5, 1:0.5})
    clf = LogisticRegression(C=1.2, fit_intercept=True, max_iter=3000,class_weight={0:0.5, 1:0.5})
    clf.fit(X=train_data.values, y=np.squeeze(train_Y))
    
    predict_train = clf.predict_proba(train_data.values)[:,1]
    predict_cv = clf.predict_proba(cv_data.values)[:,1]
    predict_test = clf.predict_proba(test_data.values)[:,1]
    
    print('训练损失:',cal_log_loss(predict_train, train_Y))
    print('测试损失:',cal_log_loss(predict_cv, cv_Y))
    
    #全量数据训练
#    train_data = pd.concat([train_data,cv_data],axis=0)
#    train_Y = np.append(train_Y, cv_Y)
#    clf.fit(X=train_data.values, y=train_Y)
#    
#    predict_test = clf.predict_proba(test_data.values)[:,1]
#    submission = pd.DataFrame({'instance_id':test_id,'predicted_score':predict_test})
#    print('预测正样本比例:',len(submission.loc[submission.predicted_score>=0.5])/len(submission))
#    submission.to_csv(r'../result/LR_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
#                  index=False, sep=' ',line_terminator='\r')
    
    