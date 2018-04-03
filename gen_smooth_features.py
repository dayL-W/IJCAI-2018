# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:12:00 2018

@author: Liaowei
"""

'''
生成用户、店铺、商品、品牌、类目在这天之前的搜索转化率平滑
'''

import pandas as pd
import numpy as np
import time
import datetime
import os
from tqdm import tqdm
from utils import raw_data_path,feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle
from smooth import BayesianSmoothing

#迭代次数和最小阈值，当二者符合其中一个条件时停止迭代
iterations = 100
eta = 0.00001


# In[]:生成用户在这天之前的CVR平滑特征
def gen_user_cvr_smooth(test_day, file_name='train'):
    
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    
    user_cvr = data.loc[data.day<test_day,['user_id', 'is_trade']]
    
    #获得每个用户的购买数和点击数
    I = user_cvr.groupby('user_id')['is_trade'].size().reset_index()
    I.columns = ['user_id', 'user_I']
    C = user_cvr.groupby('user_id')['is_trade'].sum().reset_index()
    C.columns = ['user_id', 'user_C']
    user_cvr = pd.concat([I, C['user_C']], axis=1)
    #平滑滤波
    hyper = BayesianSmoothing(1, 1)
    hyper.update(user_cvr['user_I'].values, user_cvr['user_C'].values, iterations, eta)
    alpha = hyper.alpha
    beta = hyper.beta
    user_cvr['user_cvr_smooth'] = (user_cvr['user_C'] + alpha) / (user_cvr['user_I'] + alpha + beta)
    
    #返回一个Dataframe
    return [user_cvr,alpha,beta]

# In[]:获取商品CVR平滑滤波后的数据
def gen_item_cvr_smooth(test_day, file_name='train'):
    '''
    获取商品这天之前的搜索转化率平滑
    '''
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    item_cvr = data.loc[data.day<test_day,['item_id', 'is_trade']]
    
    #获得每个商品的购买数和点击数
    I = item_cvr.groupby('item_id')['is_trade'].size().reset_index()
    I.columns = ['item_id', 'item_I']
    C = item_cvr.groupby('item_id')['is_trade'].sum().reset_index()
    C.columns = ['item_id', 'item_C']
    item_cvr = pd.concat([I, C['item_C']], axis=1)
    #平滑滤波
    hyper = BayesianSmoothing(1, 1)
    hyper.update(item_cvr['item_I'].values, item_cvr['item_C'].values, 100, 0.00001)
    alpha = hyper.alpha
    beta = hyper.beta
    item_cvr['item_cvr_smooth'] = (item_cvr['item_C'] + alpha) / (item_cvr['item_I'] + alpha + beta)
    
    return [item_cvr,alpha,beta]

def gen_brand_cvr_smooth(test_day, file_name='train'):
    '''
    获取商品品牌这天之前的搜索转化率平滑
    '''
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    brand_cvr = data.loc[data.day<test_day,['item_brand_id', 'is_trade']]
    
    #获得每个商品的购买数和点击数
    I = brand_cvr.groupby('item_brand_id')['is_trade'].size().reset_index()
    I.columns = ['item_brand_id', 'brand_I']
    C = brand_cvr.groupby('item_brand_id')['is_trade'].sum().reset_index()
    C.columns = ['item_brand_id', 'brand_C']
    brand_cvr = pd.concat([I, C['brand_C']], axis=1)
    #平滑滤波
    hyper = BayesianSmoothing(1, 1)
    hyper.update(brand_cvr['brand_I'].values, brand_cvr['brand_C'].values, 100, 0.00001)
    alpha = hyper.alpha
    beta = hyper.beta
    brand_cvr['brand_cvr_smooth'] = (brand_cvr['brand_C'] + alpha) / (brand_cvr['brand_I'] + alpha + beta)
    
    return [brand_cvr,alpha,beta]

def gen_cate_cvr_smooth(test_day, file_name='train'):
    '''
    获取商品类目这天之前的搜索转化率平滑
    '''
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    cate_cvr = data.loc[data.day<test_day,['second_cate', 'is_trade']]
    
    #获得每个商品的购买数和点击数
    I = cate_cvr.groupby('second_cate')['is_trade'].size().reset_index()
    I.columns = ['second_cate', 'cate_I']
    C = cate_cvr.groupby('second_cate')['is_trade'].sum().reset_index()
    C.columns = ['second_cate', 'cate_C']
    cate_cvr = pd.concat([I, C['cate_C']], axis=1)
    #平滑滤波
    hyper = BayesianSmoothing(1, 1)
    hyper.update(cate_cvr['cate_I'].values, cate_cvr['cate_C'].values, 100, 0.00001)
    alpha = hyper.alpha
    beta = hyper.beta
    cate_cvr['cate_cvr_smooth'] = (cate_cvr['cate_C'] + alpha) / (cate_cvr['cate_I'] + alpha + beta)
    
    return [cate_cvr,alpha, beta]

# In[]:生成店铺CVR平滑滤波后的数据
def gen_shop_cvr_smooth(test_day, file_name='train'):
    '''
    获取店铺这天之前的搜索转化率平滑
    '''
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    shop_cvr = data.loc[data.day<test_day,['shop_id', 'is_trade']]
    
    #获得每个商品的购买数和点击数
    I = shop_cvr.groupby('shop_id')['is_trade'].size().reset_index()
    I.columns = ['shop_id', 'shop_I']
    C = shop_cvr.groupby('shop_id')['is_trade'].sum().reset_index()
    C.columns = ['shop_id', 'shop_C']
    shop_cvr = pd.concat([I, C['shop_C']], axis=1)
    #平滑滤波
    hyper = BayesianSmoothing(1, 1)
    hyper.update(shop_cvr['shop_I'].values, shop_cvr['shop_C'].values, 100, 0.00001)
    alpha = hyper.alpha
    beta = hyper.beta
    shop_cvr['shop_cvr_smooth'] = (shop_cvr['shop_C'] + alpha) / (shop_cvr['shop_I'] + alpha + beta)
    
    return [shop_cvr,alpha, beta]

# In[]
def gen_col_cvr_stats(x, col_cvr):
    """
    统计cate-property对的转化率，取最大值，最小值，均值
    """
    second_item_cate = x['item_category_list'].split(';')[1]
    properties = x['item_property_list'].split(';')
    cvr_list = []
    for prop in properties:
        cate_prop_col = second_item_cate+'_'+prop
        if cate_prop_col in col_cvr.keys():
            cvr_list.append(col_cvr[cate_prop_col])
    if len(cvr_list) == 0:
        return [-1,-1,-1]
    return [max(cvr_list), min(cvr_list), np.mean(cvr_list)]

def gen_cvr_smooth(file_name='train'):
    
    data = load_pickle(path=raw_data_path + file_name + '.pkl')
    cols = ['user_id','item_id', 'item_brand_id', 'second_cate', 'shop_id']
    
    if file_name == 'train':
        #对每个特征
        feat_all = None
        for col in cols:
            #对于每一天
            col_feat_all = None
            for day in (data.day).unique():
                #筛选出这天之前的数据
                col_cvr_dict = dict()
                col_I = col+'_I'
                col_C = col+'_C'
                col_cvr_smooth = col+'_cvr_smooth'
                print(day)
                if day != 17:
                    filter_data = data.loc[data.day < day, [col, 'is_trade']]
                    #计算转化率
                    I = filter_data.groupby(col)['is_trade'].size().reset_index()
                    I.columns = [col, col_I]
                    C = filter_data.groupby(col)['is_trade'].sum().reset_index()
                    C.columns = [col, col_C]
                    col_cvr = pd.concat([I, C[col_C]], axis=1)
                    #平滑滤波
                    hyper = BayesianSmoothing(1, 1)
                    hyper.update(col_cvr[col_I].values, col_cvr[col_C].values, 100, 0.00001)
                    alpha = hyper.alpha
                    beta = hyper.beta
                    col_cvr[col_cvr_smooth] = (col_cvr[col_C] + alpha) / (col_cvr[col_I] + alpha + beta)
#                    col_cvr_dict = dict(col_cvr[[col, col_cvr_smooth]].values)
                    col_cvr_series = col_cvr[[col, col_cvr_smooth]].set_index(col)[col_cvr_smooth]
                    
                    #把今天之前的转化率加到今天的特征中
                col_feat = data.loc[data.day==day, ['instance_id', col]]
#                col_feat[col_cvr_smooth] = col_feat.apply(lambda x: col_cvr_dict[x[col]] if x[col] in col_cvr_dict.keys() else -1, axis=1)
                col_feat[col_cvr_smooth] = col_feat.apply(lambda x: col_cvr_series[x[col]] if x[col] in col_cvr_series.index else -1, axis=1)
                col_feat_all = pd.concat([col_feat_all,col_feat], axis=0)
            #保存数据
            feat_all = pd.concat([feat_all,col_feat_all[col_cvr_smooth]], axis=1)
            cvr_path = feature_data_path + 'train_cvr_smooth'
            dump_pickle(feat_all, cvr_path)
    else:
        train_data = load_pickle(path=raw_data_path + 'train' + '.pkl')
        #对每个特征
        feat_all = None
        for col in cols:
            #筛选出这天之前的数据
            col_cvr_dict = dict()
            col_I = col+'_I'
            col_C = col+'_C'
            col_cvr_smooth = col+'_cvr_smooth'
            
            filter_data = train_data.loc[:, [col, 'is_trade']]
            #计算转化率
            I = filter_data.groupby(col)['is_trade'].size().reset_index()
            I.columns = [col, col_I]
            C = filter_data.groupby(col)['is_trade'].sum().reset_index()
            C.columns = [col, col_C]
            col_cvr = pd.concat([I, C[col_C]], axis=1)
            #平滑滤波
            hyper = BayesianSmoothing(1, 1)
            hyper.update(col_cvr[col_I].values, col_cvr[col_C].values, 100, 0.00001)
            alpha = hyper.alpha
            beta = hyper.beta
            col_cvr[col_cvr_smooth] = (col_cvr[col_C] + alpha) / (col_cvr[col_I] + alpha + beta)
#            col_cvr_dict = dict(col_cvr[[col, col_cvr_smooth]].values)
            col_cvr_series = col_cvr[[col, col_cvr_smooth]].set_index(col)[col_cvr_smooth]
                
                #把今天之前的转化率加到今天的特征中
            col_feat = data.loc[:, ['instance_id', col]]
#            col_feat[col_cvr_smooth] = col_feat.apply(lambda x: col_cvr_dict[x[col]] if x[col] in col_cvr_dict.keys() else -1, axis=1)
            col_feat[col_cvr_smooth] = col_feat.apply(lambda x: col_cvr_series[x[col]] if x[col] in col_cvr_series.index else -1, axis=1)
            feat_all = pd.concat([feat_all,col_feat[col_cvr_smooth]], axis=1)
        #保存数据
        cvr_path = feature_data_path + 'test_cvr_smooth'
        dump_pickle(feat_all, cvr_path)
# In[]
if __name__ == '__main__':
    gen_cvr_smooth('train')
    gen_cvr_smooth('test')


