# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:54:37 2018

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

import xgboost as xgb
import operator 
import matplotlib.pyplot as plt


def ceate_feature_map(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1
#        print(i)
    outfile.close() 
    
params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric':'logloss',
    'max_depth':8,
    'colsample_bytree': 1,
#    'lambda':1,
#    'eta':0.2,
#    'silent':0,
#    'alpha':3,
#    'subsample':1,
}
n_round=20


if __name__ == '__main__':
    
    t0 = time.time()
    train_data = load_pickle(path=cache_pkl_path +'train_data')
    train_Y = load_pickle(path=cache_pkl_path +'train_Y')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data')
    cv_Y = load_pickle(path=cache_pkl_path +'cv_Y')
    
    test_data = load_pickle(path=cache_pkl_path +'test_data')
    test_file = 'round1_ijcai_18_test_a_20180301.txt'
    test = pd.read_table(raw_data_path + test_file,delim_whitespace=True)
    test_id = test.instance_id
    
    train_data.drop(['user_id','shop_id','item_id','item_brand_id'],axis=1,inplace=True)
    cv_data.drop(['user_id','shop_id','item_id','item_brand_id'],axis=1,inplace=True)
    test_data.drop(['user_id','shop_id','item_id','item_brand_id'],axis=1,inplace=True)
    
    print('train shap:',train_data.shape)
    print('cv shape', cv_data.shape)
    print('test shape', test_data.shape)

    
    train_feat = xgb.DMatrix(train_data.values, label=train_Y)
    cv_feat = xgb.DMatrix(cv_data.values)
    test_feat = xgb.DMatrix(test_data.values)

    clf = xgb.train(params=params, dtrain=train_feat,num_boost_round=n_round)
    
    predict_train = clf.predict(train_feat)
    predict_cv = clf.predict(cv_feat)
    predict_test = clf.predict(test_feat)
    
    #查看特征重要度
    features = train_data.columns
    ceate_feature_map(features) 
    
    importance = clf.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
    df['fscore'] = df['fscore'] / df['fscore'].sum()

#    plt.figure()
#    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
#    plt.title('XGBoost Feature Importance')  
#    plt.xlabel('relative importance')  
#    plt.show()

    print('训练损失:',cal_log_loss(predict_train, train_Y))
    print('测试损失:',cal_log_loss(predict_cv, cv_Y))
    
    t1 = time.time()
    print('训练用时:',t1-t0)
    
    submission = pd.DataFrame({'instance_id':test_id,'predicted_score':predict_test})
    
    submission.to_csv(r'../result/xgb_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, sep=' ',line_terminator='\r')