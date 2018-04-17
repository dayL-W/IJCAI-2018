# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:13:45 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
from utils import load_pickle, raw_data_path, feature_data_path, cache_pkl_path, result_path, model_path,submmit_result

if __name__ == '__main__':
    
    
    LR = pd.read_csv('../result/LR_20180416_175004.txt',sep=' ')
    XGB = pd.read_csv('../result/XGB_20180416_213941.txt',sep=' ')
    LGB = pd.read_csv('../result/LGB_20180416_221018.txt',sep=' ')
    FFM = pd.read_csv('../result/FFM_20180416_222912.txt',sep=' ')
    
    result = np.zeros((len(LR), 4))
    result[:,0] = LR['predicted_score'].values
    result[:,1] = XGB['predicted_score'].values
    result[:,2] = LGB['predicted_score'].values
    result[:,3] = FFM['predicted_score'].values
    median = np.median(result, axis=1)
    
    submmit_result(median, 'median')