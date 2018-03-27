# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:53:53 2018

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
from sklearn.preprocessing import OneHotEncoder  
from sklearn.ensemble import GradientBoostingClassifier 

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
    
    train_data.drop(['user_id','shop_id','item_id','item_brand_id'],axis=1,inplace=True)
    cv_data.drop(['user_id','shop_id','item_id','item_brand_id'],axis=1,inplace=True)
    test_data.drop(['user_id','shop_id','item_id','item_brand_id'],axis=1,inplace=True)
    
    print('train shap:',train_data.shape)
    print('cv shape', cv_data.shape)
    print('test shape', test_data.shape)
    
    gbc = GradientBoostingClassifier(n_estimators=20,max_depth=5,max_leaf_nodes=12)
    gbc.fit(train_data.values, train_Y)
    predict_train = gbc.predict_proba(train_data.values)[:,1]
    predict_cv = gbc.predict_proba(cv_data.values)[:,1]
    predict_test = gbc.predict_proba(test_data.values)[:,1]
    
    print('训练损失:',cal_log_loss(predict_train, train_Y))
    print('测试损失:',cal_log_loss(predict_cv, cv_Y))
    
    
    #构造新的训练和测试集
    x = gbc.apply(train_data)
    x = x.reshape((x.shape[0],x.shape[1]))
    enc = OneHotEncoder()
    enc.fit(x)
    new_feature_train=enc.transform(x)
    new_feature_train=new_feature_train.toarray()
    new_train=np.concatenate([train_data.values,new_feature_train],axis=1)
    
    x = gbc.apply(cv_data.values)
    x = x.reshape((x.shape[0],x.shape[1]))
    new_feature_cv=enc.transform(x)
    new_feature_cv=new_feature_cv.toarray()
    new_cv=np.concatenate([cv_data.values,new_feature_cv],axis=1)
    
    x = gbc.apply(test_data.values)
    x = x.reshape((x.shape[0],x.shape[1]))
    new_feature_test=enc.transform(x)
    new_feature_test=new_feature_test.toarray()
    new_test=np.concatenate([test_data.values,new_feature_test],axis=1)
    
    print('train shap:',new_train.shape)
    print('cv shape', new_cv.shape)
    print('test shape', new_test.shape)
    
    #LR预测
    clf = LogisticRegression(C=0.8, fit_intercept=True, max_iter=3000,class_weight={0:0.5, 1:0.5})
    clf.fit(X=new_train, y=np.squeeze(train_Y))
    
    predict_train = clf.predict_proba(new_train)[:,1]
    predict_cv = clf.predict_proba(new_cv)[:,1]
    predict_test = clf.predict_proba(new_test)[:,1]
    
    print('训练损失:',cal_log_loss(predict_train, train_Y))
    print('测试损失:',cal_log_loss(predict_cv, cv_Y))
    t1 = time.time()
    print('训练用时:',t1-t0)
    
    submission = pd.DataFrame({'instance_id':test_id,'predicted_score':predict_test})
    print('预测正样本比例:',len(submission.loc[submission.predicted_score>=0.5])/len(submission))
    submission.to_csv(r'../result/GBDT_LR_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, sep=' ',line_terminator='\r')