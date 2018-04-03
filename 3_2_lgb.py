# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:51:11 2018

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

import lightgbm as lgb

params = {
    'max_depth': 4,                 #4
#    'min_data_in_leaf': 40,-
    'feature_fraction': 1,       #1
    'learning_rate': 0.04,          #0.04
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'verbose': -1,
    'metric': 'binary_logloss',
}

if __name__ == '__main__':
    
    
    t0 = time.time()
    train_data = load_pickle(path=cache_pkl_path +'train_data')
    train_Y = load_pickle(path=cache_pkl_path +'train_Y')
#    new_cvr = load_pickle(path=cache_pkl_path +'train_data_cvr_fusion')
#    train_data['new_cvr'] = new_cvr.values
    
    cv_data = load_pickle(path=cache_pkl_path +'cv_data')
    cv_Y = load_pickle(path=cache_pkl_path +'cv_Y')
#    new_cvr = load_pickle(path=cache_pkl_path +'cv_data_cvr_fusion')
#    cv_data['new_cvr'] = new_cvr.values
    
    test_data = load_pickle(path=cache_pkl_path +'test_data')
#    new_cvr = load_pickle(path=cache_pkl_path +'test_data_cvr_fusion')
#    test_data['new_cvr'] = new_cvr.values
    
    test_file = 'round1_ijcai_18_test_a_20180301.txt'
    test = pd.read_table(raw_data_path + test_file,delim_whitespace=True)
    test_id = test.instance_id
    
    drop_cols = ['user_id','shop_id','item_id','item_brand_id']
    train_data.drop(drop_cols,axis=1,inplace=True)
    cv_data.drop(drop_cols,axis=1,inplace=True)
    test_data.drop(drop_cols,axis=1,inplace=True)
    
    print('train shap:',train_data.shape)
    print('cv shape', cv_data.shape)
    print('test shape', test_data.shape)
    
    lgb_train = lgb.Dataset(train_data.values, train_Y)
    lgb_cv = lgb.Dataset(cv_data.values, cv_Y)
    gbm = lgb.train(params=params,            #参数
                    train_set=lgb_train,      #要训练的数据
                    num_boost_round=6000,     #迭代次数
                    valid_sets=lgb_cv,        #训练时需要评估的列表
                    verbose_eval=False,       #
                    
                    early_stopping_rounds=200)
    
    predict_train = gbm.predict(train_data.values)
    predict_cv = gbm.predict(cv_data.values)
    predict_test = gbm.predict(test_data.values)
    
    feat_imp = pd.Series(gbm.feature_importance(), index=train_data.columns).sort_values(ascending=False)

    print('训练损失:',cal_log_loss(predict_train, train_Y))
    print('测试损失:',cal_log_loss(predict_cv, cv_Y))
    t1 = time.time()
    print('训练时间:',t1 - t0)
    
    #全量评测
    train_data = pd.concat([train_data, cv_data],axis=0)
    train_Y = np.append(train_Y, cv_Y)
    
    lgb_train = lgb.Dataset(train_data.values, train_Y)
    gbm = lgb.train(params=params,            #参数
                    train_set=lgb_train,      #要训练的数据
                    num_boost_round=300,     #迭代次数
                    verbose_eval=True)
    predict_test = gbm.predict(test_data.values)
    print('训练损失:',cal_log_loss(gbm.predict(train_data.values), train_Y))
    
    submission = pd.DataFrame({'instance_id':test_id,'predicted_score':predict_test})
    print('预测正样本比例:',len(submission.loc[submission.predicted_score>=0.5])/len(submission))
    submission.to_csv(r'../result/lgb_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, sep=' ',line_terminator='\r')
    
    
    