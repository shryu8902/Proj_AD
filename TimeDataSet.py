#%%
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os, glob, json, pickle
import pandas as pd
import numpy as np

class TimeseriesAnomalyDataset(torch.utils.data.Dataset): 
    def __init__(self, TARGET_DATA, samprate, window, stride, json_root, scaler='MMSC', ABNORM_TRJ_CLASS=[], ABNORM_STEP_CLASS=[]):
        self.TARGET_DATA=TARGET_DATA
        with open(json_root,'rb') as f:
            DataInfo=json.load(f)
        self._ROOT_ = DataInfo['ROOT'] + TARGET_DATA
        self.DIM = DataInfo[TARGET_DATA]['TRJ_CLASS']
        self.TRJ_CLASS = DataInfo[TARGET_DATA]['TRJ_CLASS']
        self.STEP_CLASS = DataInfo[TARGET_DATA]['STEP_CLASS']
        self.NORM_TRJ_CLASS = [item for item in self.TRJ_CLASS if item not in ABNORM_TRJ_CLASS]
        self.ABNORM_TRJ_CLASS = ABNORM_TRJ_CLASS
        self.NORM_STEP_CLASS = [item for item in self.STEP_CLASS if item not in ABNORM_STEP_CLASS]
        self.ABNORM_STEP_CLASS = ABNORM_STEP_CLASS
        self.WINDOW = window
        self.SAMPRATE = samprate
        self.STRIDE = stride

        if scaler is not None:
            with open(self._ROOT_+'/scaler.pkl','rb') as f: 
                self.SCALER = pickle.load(f)[scaler]
        else:
            self.SCALER = None

        self.file_list=[]
        for i in self.NORM_TRJ_CLASS:
            self.file_list += glob.glob(self._ROOT_+f'/{i}/*')

        # Create list of sampled (with sampling rate of self.SAMPRATE) data
        self.X_DATA = []
        X_NORMAL = []
        for i in self.file_list:
            with open(i,'rb') as f:
                temp=pickle.load(f)[::self.SAMPRATE] #self.SAMPRATE
                temp.reset_index(drop=True,inplace=True)
                if self.SCALER is not None:
                    scaled_temp = self.SCALER.transform(np.array(temp)[:,:-1])                
                else:
                    scaled_temp = np.array(temp)[:,:-1]    
                self.X_DATA.append(scaled_temp) 
                normal_temp=temp[temp.cls.isin(self.NORM_STEP_CLASS)]
                X_NORMAL.append(normal_temp[normal_temp.index>=(self.WINDOW-1)][::self.STRIDE].cls) 
        self.NORM_STEP_CLASS_INFO = pd.concat(X_NORMAL,keys=range(len(X_NORMAL))).reset_index()    

    def __len__(self):
        return(len(self.NORM_STEP_CLASS_INFO))
    
    def __getitem__(self, idx): 
        tr_idx, step_idx, step_cls = self.NORM_STEP_CLASS_INFO.loc[idx] 
        return torch.Tensor(self.X_DATA[tr_idx][step_idx-self.WINDOW+1:step_idx,:]), torch.Tensor([step_cls])
