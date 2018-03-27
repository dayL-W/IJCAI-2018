# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:46:04 2018

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

 # In[]:读取特征并整合成训练数据和测试数据
def gen_train_data(file_name='train', test_day=24):
    
    data = pd.DataFrame()
    
    #读取做好的特征文件
    user_basic_info = load_pickle(path=feature_data_path + file_name + '_user_basic_info')
    user_search_count = load_pickle(path=feature_data_path + file_name + '_user_search_count')
    user_search_time = load_pickle(path=feature_data_path + file_name + '_user_search_time')
    
    item_basic_info = load_pickle(path=feature_data_path + file_name + '_item_basic_info')
    item_relative_info = load_pickle(path=feature_data_path + file_name + '_item_relative_feature')
    query_item_sim = load_pickle(path=feature_data_path + file_name + '_query_item_sim')
    
    shop_basic_info = load_pickle(path=feature_data_path + file_name + '_shop_basic_info')
    
    buy_count = load_pickle(path=feature_data_path + file_name + '_buy_count')
    cvr_smooth = load_pickle(path=feature_data_path + file_name + '_cvr_smooth')
    cate_prop_cvr = load_pickle(path=feature_data_path + file_name + '_cate_prop_cvr')
    data = pd.concat([user_basic_info,user_search_count,user_search_time,\
                      item_basic_info,item_relative_info,query_item_sim,shop_basic_info,\
                      buy_count,cvr_smooth,cate_prop_cvr],axis=1)
    
    #把销量、价格、收藏次数以下特征取对数
    data['item_sales_level'].replace(to_replace=-1,value=0,inplace=True)
    cols = ['item_sales_level','item_collected_level','item_pv_level']
    for col in cols:
        data[col] = np.log1p(data[col])
        
    if file_name == 'train':
        #划分训练数据和测试数据
        train_data = data.loc[data.day<test_day,:]
        cv_data = data.loc[data.day>=test_day,:]
        
        #对训练数据的负样本进行1/7的采样
#        train_data = build_train_dataset(train_data)
        train_Y = train_data.is_trade.values
        train_data.drop(['is_trade'],axis=1,inplace=True)
        
        test_Y = cv_data.is_trade.values
        cv_data.drop(['is_trade'],axis=1,inplace=True)
        
        cv_data.reset_index(inplace=True,drop=True)
        train_data.reset_index(inplace=True,drop=True)
        #保存文件
        dump_pickle(train_data, cache_pkl_path +'train_data')
        dump_pickle(train_Y, cache_pkl_path +'train_Y')
        dump_pickle(cv_data, cache_pkl_path +'cv_data')
        dump_pickle(test_Y, cache_pkl_path +'cv_Y')
    else:
        data.reset_index(inplace=True,drop=True)
        dump_pickle(data, cache_pkl_path +'test_data')
    
def gen_one_hot_data():
    train_data = load_pickle(path=cache_pkl_path +'train_data')
    cv_data = load_pickle(path=cache_pkl_path +'cv_data')
    test_data= load_pickle(path=cache_pkl_path +'test_data')
    
    cv_data.index += len(train_data)
    test_data.index += len(train_data) + len(cv_data)
    
    data = pd.concat([train_data,cv_data, test_data],axis=0)
    
#    cols = ['user_gender_id','user_age_level','user_occupation_id','user_star_level'\
#            ,'second_cate','item_city_id','item_price_level','item_sales_level'\
#            ,'item_collected_level','item_pv_level','context_page_id','shop_review_num_level'\
#            ,'shop_star_level']
    
    cols = ['user_gender_id','user_age_level','user_occupation_id'
            ,'second_cate','item_city_id','item_price_level'
            ,'context_page_id','shop_review_num_level']
    
    for col in cols:
        col_feature = pd.get_dummies(data[col], prefix=col)
        data.drop([col],axis=1,inplace=True)
        data = pd.concat([data,col_feature], axis=1)
                                       
#    data.drop(['item_cvr_smooth','user_cvr_smooth','shop_cvr_smooth','brand_cvr_smooth',\
#              'cate_cvr_smooth','user_buy_count','item_buy_count','brand_buy_count','cate_buy_count',\
#               'shop_buy_count','user_I','item_I','brand_I','cate_I','shop_I'],axis=1,inplace=True)

    X = minmax_scale(data.values)
    data = pd.DataFrame(data=X, columns=data.columns)
    
    train_data = data.loc[train_data.index]
    cv_data = data.loc[cv_data.index]
    test_data = data.loc[test_data.index]
    
    
    train_data.reset_index(inplace=True,drop=True)
    cv_data.reset_index(inplace=True,drop=True)
    test_data.reset_index(inplace=True,drop=True)
    
    dump_pickle(train_data, cache_pkl_path +'train_data_onehot')
    dump_pickle(cv_data, cache_pkl_path +'cv_data_onehot')
    dump_pickle(test_data, cache_pkl_path +'test_data_onehot')
    
    
if __name__ == '__main__':
    
    gen_train_data('train')
    gen_train_data('test')
    gen_one_hot_data()