# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:13:45 2018

@author: Liaowei
"""

import pandas as pd
import numpy as np
from utils import load_pickle, raw_data_path, feature_data_path, cache_pkl_path, result_path, model_path,submmit_result

if __name__ == '__main__':
    
    
    XGB = pd.read_csv('../result/XGB_20180417_212149.txt',sep=' ')
    LGB = pd.read_csv('../result/LGB_20180417_160159.txt',sep=' ')
    FFM = pd.read_csv('../result/FFM_20180417_150554.txt',sep=' ')
    
    result = np.zeros((len(XGB), 3))
    result[:,0] = XGB['predicted_score'].values
    result[:,1] = LGB['predicted_score'].values
    result[:,2] = FFM['predicted_score'].values
    median = np.mean(result, axis=1)
    
    submmit_result(median, 'median')