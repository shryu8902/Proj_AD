#%%
import os, glob, json, pickle
import pandas as pd
import numpy as np

#torch related
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

#pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

#logging
import wandb
from pytorch_lightning.loggers import WandbLogger

# import custom class for time series dataset
# from TimeDataSet import TimeseriesAnomalyDataset

# others
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# from QAE import QuantileLitAutoEncoder
#%% 
# Autoencoder and LSTM autoencoder example with pytorch lightning
class LitAutoEncoder(pl.LightningModule):
    """
    Autoencoder :
        Z = f(X)   
        X_hat = g(Z)

    LSTM Autoencoder :
        z_t = f(x_t, h_t-1)
        x_hat_t = g(z_t,h'_t-1)
    
    """

    def __init__(self, window, dim, latdim):
        super().__init__()
        # initialize network structure
        self.window = window
        self.dim = dim
        self.latdim=latdim

        # log hyperparameters
        self.save_hyperparameters()
        # autoencoder example
        self.encoder = nn.Sequential(
            nn.Linear(self.window*self.dim,512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.latdim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latdim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.window*self.dim),
        )
        # LSTM autoencoder example
        self.en_LSTM = nn.LSTM(input_size = self.dim, hidden_size = 64, num_layers = 2, batch_first = True, bidirectional=True)
        self.en_lin = nn.Sequential(
            nn.Linear(64*2,self.latdim),
            nn.LeakyReLU()
        )
        self.de_LSTM = nn.LSTM(input_size = self.latdim, hidden_size = 64, num_layers = 2, batch_first = True, bidirectional=True)
        self.de_lin = nn.Sequential(
            nn.Linear(64*2,self.dim),
        )

    def anomaly_score(self, x_hat, x):
        # Define anomaly score metric
        # as MSE between raw vs predicted.
        # or you can define own anomaly score inside forward step. e.g., rapp.
        score = F.mse_loss(x_hat,x,reduction='none')
        if len(x.shape)==3:
            return score.mean(axis=(1,2)).detach().numpy()
        elif len(x.shape)==2:
            return score.mean(axis=(1)).detach().numpy()
        return score

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # forward step for calculating anomaly score

        # Autoencoder
        # x = x.view(x.size(0), -1)
        # embedding = self.encoder(x)
        # x_hat = self.decoder(embedding)

        # LSTM Autoencoder
        z1,_ = self.en_LSTM(x)
        embedding = self.en_lin(z1)
        x_hat1,_ = self.de_LSTM(embedding)
        x_hat = self.de_lin(x_hat1)

        # define anomaly score as mse between raw vs predicted.
        AnomalyScore = self.anomaly_score(x_hat, x)
        return x_hat.detach(), AnomalyScore

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Autoencoder
        # x = x.view(x.size(0), -1)
        # embedding = self.encoder(x)
        # x_hat = self.decoder(embedding)

        # LSTM Autoencoder
        z1,_ = self.en_LSTM(x)
        embedding = self.en_lin(z1)
        x_hat1,_ = self.de_LSTM(embedding)
        x_hat = self.de_lin(x_hat1)

        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Autoencoder
        # x = x.view(x.size(0), -1)
        # embedding = self.encoder(x)
        # x_hat = self.decoder(embedding)

        # LSTM Autoencoder
        z1,_ = self.en_LSTM(x)
        embedding = self.en_lin(z1)
        x_hat1,_ = self.de_LSTM(embedding)
        x_hat = self.de_lin(x_hat1)
        loss = F.mse_loss(x_hat, x)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # Autoencoder
        # x = x.view(x.size(0), -1)
        # embedding = self.encoder(x)
        # x_hat = self.decoder(embedding)

        # LSTM Autoencoder
        z1,_ = self.en_LSTM(x)
        embedding = self.en_lin(z1)
        x_hat1,_ = self.de_LSTM(embedding)
        x_hat = self.de_lin(x_hat1)

        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor":"valid_loss"}
        #without lr scheduler           
        # return optimizer

#%% 
""" Leave one-out Anomaly Detection 
    0 NO_LABEL
    1 WALKING           
    2 WALKING_UPSTAIRS  
    3 WALKING_DOWNSTAIRS
    4 SITTING           
    5 STANDING          
    6 LAYING            
    7 STAND_TO_SIT      
    8 SIT_TO_STAND      
    9 SIT_TO_LIE        
    10 LIE_TO_SIT        
    11 STAND_TO_LIE      
    12 LIE_TO_STAND      
"""
# Random Settings
pl.seed_everything(1234)

# AD Settings : 
TARGET = 'HAR'
AB_CLASS = [1]
WINDOW = 128
SAMPRATE = 2
STRIDE = 30
SCALE = 'STSC' # 'STSC' : Standard scaler, 'MMSC' : Min-max scaller, None : No normalizing 
Z_DIM = 2

with open('./DataInfo.JSON','rb') as f:
    DataInfo=json.load(f)
STEP_CLASS = DataInfo[TARGET]['STEP_CLASS']
NORM_CLASS = [x for x in STEP_CLASS if x not in AB_CLASS ]
DIM = DataInfo[TARGET]['DIM']

# Create custom dataset class from pre-processed data
dataset = TimeseriesAnomalyDataset('HAR',samprate=SAMPRATE, window=WINDOW, stride = STRIDE, scaler=SCALE,  
            json_root = './DataInfo.JSON', ABNORM_STEP_CLASS=AB_CLASS,timestamp=False,fileformat='pkl',transient=True)
dataset_ab = TimeseriesAnomalyDataset('HAR',samprate=SAMPRATE, window=WINDOW, stride = STRIDE, scaler=SCALE,
            json_root = './DataInfo.JSON', ABNORM_STEP_CLASS=NORM_CLASS,timestamp=False,fileformat='pkl',transient=True)

# Split data into train, val, test_in (normal samples), test_out (abnormal samples)
# n_test_in == n_test_out, n_train = 80%, n_val = 20%
n_normal = len(dataset)
n_ab = len(dataset_ab)
n_train = int(0.8*(n_normal-n_ab))
n_val = n_normal-n_ab-n_train
train_, test = random_split(dataset, [n_normal-n_ab, n_ab])
train, val = random_split(train_, [n_train,n_val])

# define loaders
train_loader = DataLoader(train,256,True,num_workers=16) # with random batch
val_loader = DataLoader(val,256,num_workers=16)
test_in_loader = DataLoader(test,256,num_workers=16)
test_out_loader = DataLoader(dataset_ab,256,num_workers=16)

# create model
autoencoder = QuantileLitAutoEncoder(window=WINDOW, dim=DIM, latdim=Z_DIM)
autoencoder_r = LitAutoEncoder(window=WINDOW, dim=DIM, latdim=Z_DIM)
autoencoder_q = QuantileLitAutoEncoder(window=WINDOW, dim=DIM, latdim=Z_DIM,quantiles=[0.3,0.5,0.7])

# Define Callbacks e.g., early stopping
# early_stopping = EarlyStopping('valid_loss')
# train model
# trainer = pl.Trainer(gpus=[2], max_epochs=100,callbacks=[early_stopping])
trainer = pl.Trainer(gpus=[3], max_epochs=50)
trainer.fit(autoencoder, train_loader,val_dataloaders=val_loader)

trainer2 = pl.Trainer(gpus=[4],max_epochs=50)
trainer2.fit(autoencoder_r, train_loader, val_dataloaders= val_loader)

trainer = pl.Trainer(gpus=[3], max_epochs=50)
trainer.fit(autoencoder_q, train_loader,val_dataloaders=val_loader)

for step, (x,y) in enumerate(val_loader):
    print(step)
trainer.test(autoencoder, test_in_loader)
autoencoder.test(test_in_loader)
autoencoder(x)
#%%
# Analysis
scores = []
mse_score = []
rmse_score = []
var_score = []
label = []
for loader in [test_in_loader, test_out_loader]:
    for step, (x,y) in enumerate(loader):
        print(step, x.sum(), y.sum())
        [a,x_hat,b],temp_score = autoencoder(x.cuda())
        scores.append(temp_score)
        temp_var_score = ((x_hat-a)+(b-x_hat)).mean(axis=(1,2))
        var_score.append(temp_var_score)
        temp_rmse = (((x_hat-x.cuda())**2)**(1/2)).mean(axis=(1,2))
        temp_mse = ((x_hat-x.cuda())**2).mean(axis=(1,2))
        rmse_score.append(temp_rmse)
        mse_score.append(temp_mse)
        y=y.numpy().reshape(-1)
        temp_label = np.array([1 if x in AB_CLASS else 0 for x in y])
        label.append(temp_label)
scores = [x.cpu() for x in scores]
mse_score = [x.cpu() for x in mse_score]
rmse_score = [x.cpu() for x in rmse_score]
var_score = [x.cpu() for x in var_score]
scores = np.concatenate(scores)
mse_score = np.concatenate(mse_score)
rmse_score = np.concatenate(rmse_score)
var_score = np.concatenate(var_score)
label = np.concatenate(label)
#%%
scores_q = []
mse_score_q = []
var_score_q = []
label_q = []
for loader in [test_in_loader, test_out_loader]:
    for step, (x,y) in enumerate(loader):
        print(step, x.sum(), y.sum())
        [a,x_hat,b],temp_score = autoencoder_q(x)
        scores_q.append(temp_score)
        temp_var_score = ((x_hat-a)+(b-x_hat)).mean(axis=(1,2))
        var_score_q.append(temp_var_score)
        temp_mse = (((x_hat-x)**2)**(1/2)).mean(axis=(1,2))
        mse_score_q.append(temp_mse)
        y=y.numpy().reshape(-1)
        temp_label = np.array([1 if x in AB_CLASS else 0 for x in y])
        label_q.append(temp_label)
scores_q = [x.cpu() for x in scores_q]
mse_score_q = [x.cpu() for x in mse_score_q]
var_score_q = [x.cpu() for x in var_score_q]
scores_q = np.concatenate(scores_q)
mse_score_q = np.concatenate(mse_score_q)
var_score_q = np.concatenate(var_score_q)
label_q = np.concatenate(label_q)

#%%
scores2 = []
label2 = []
for loader in [test_in_loader, test_out_loader]:
    for step, (x,y) in enumerate(loader):
        print(step, x.sum(), y.sum())
        x_hat,temp_score = autoencoder2(x)
        scores2.append(temp_score)
        y=y.numpy().reshape(-1)
        temp_label = np.array([1 if x in AB_CLASS else 0 for x in y])
        label2.append(temp_label)
scores2 = np.concatenate(scores2)
label2 = np.concatenate(label2)

#%%

# calculate ROC
fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1)
fpr2, tpr2, thresholds2 = roc_curve(label, mse_score, pos_label=1)
fpr2_r, tpr2_r, thresholds2_r = roc_curve(label, rmse_score, pos_label=1)

fpr3, tpr3, thresholds3 = roc_curve(label, var_score, pos_label=1)

fpr4, tpr4, thresholds4 = roc_curve(label, (var_score+mse_score)/2, pos_label=1)
fpr_q, tpr_q, thresholds_q = roc_curve(label_q, scores_q, pos_label=1)
fpr2_q, tpr2_q, thresholds2_q = roc_curve(label_q, mse_score_q**2, pos_label=1)
fpr3_q, tpr3_q, thresholds3_q = roc_curve(label_q, var_score_q, pos_label=1)
# fpr4, tpr4, thresholds4 = roc_curve(label, (var_score+mse_score_q)/2, pos_label=1)


fpr5, tpr5, thresholds5 = roc_curve(label2, scores2, pos_label=1)
fpr6, tpr6, thresholds6 = roc_curve(label2, np.sqrt(scores2), pos_label=1)

roc_auc = auc(fpr,tpr)
roc_auc2 = auc(fpr2,tpr2)
roc_auc2_r = auc(fpr2_r,tpr2_r)

roc_auc3 = auc(fpr3,tpr3)
roc_auc_q = auc(fpr_q,tpr_q)
roc_auc2_q = auc(fpr2_q,tpr2_q)
roc_auc3_q = auc(fpr3_q,tpr3_q)


roc_auc4 = auc(fpr4,tpr4)
roc_auc5 = auc(fpr5,tpr5)
roc_auc6 = auc(fpr6,tpr6)

plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='QVAE (AUC = %0.4f)' % roc_auc)
plt.plot(fpr2, tpr2, color='blue',
         lw=2, label='VAE (AUC = %0.4f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='darkgreen',
         lw=2, label='QV (AUC = %0.4f)' % roc_auc3)
plt.plot(fpr_q, tpr_q, 
         lw=2, label='QVAE37 (AUC = %0.4f)' % roc_auc_q)
plt.plot(fpr2_q, tpr2_q, 
         lw=2, label='VAE37 (AUC = %0.4f)' % roc_auc2_q)
plt.plot(fpr3_q, tpr3_q, 
         lw=2, label='QV37 (AUC = %0.4f)' % roc_auc3_q)

# plt.plot(fpr4, tpr4, color='red',
#          lw=2, label='QV (AUC = %0.4f)' % roc_auc4)
# plt.plot(fpr5, tpr5, color='black',
#          lw=2, label='QV (AUC = %0.4f)' % roc_auc5)
# plt.plot(fpr6, tpr6, color='purple',
#          lw=2, label='QV (AUC = %0.4f)' % roc_auc6)

plt.legend()
plt.grid()
#%%
plt.plot(scores_q)
# plt.plot(mse_score)
plt.plot(label_q)


#%%
STEP_CLASS = DataInfo[TARGET]['STEP_CLASS']

dataset_list = []
for i in range(13):
    NORM_CLASS = [i]
    AB_CLASS = [x for x in STEP_CLASS if x not in NORM_CLASS]
    dataset =TimeseriesAnomalyDataset('HAR',samprate=SAMPRATE, window=WINDOW, stride = STRIDE, scaler=SCALE,
            json_root = './DataInfo.JSON', ABNORM_STEP_CLASS=AB_CLASS) 
    dataset_list.append(dataset)

# %%
for ind, i in enumerate(dataset_list):
    # print(len(i))
    summation = 0

    for j in range(len(i)):
        summation+=i.__getitem__(j)[0].sum().numpy()
    print(ind, summation)
# %%
