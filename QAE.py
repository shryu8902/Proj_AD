#%%
import torch
from torch import nn
import torch.nn.functional as F

#pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from QuantileLoss import QuantileLoss
import numpy as np

class QuantileLitAutoEncoder(pl.LightningModule):
    """
    Autoencoder :
        Z = f(X)   
        X_hat = g(Z)

    LSTM Autoencoder :
        z_t = f(x_t, h_t-1)
        x_hat_t = g(z_t,h'_t-1)
    
    """

    def __init__(self, window, dim, latdim, quantiles=[0.1,0.5,0.9]):
        super().__init__()
        # initialize network structure
        self.window = window
        self.dim = dim
        self.latdim=latdim
        self.quantiles=quantiles
        self.Qloss = QuantileLoss(quantiles=self.quantiles, reduction='sum')
        # log hyperparameters
        self.save_hyperparameters()
        # LSTM autoencoder example
        self.en_LSTM = nn.LSTM(input_size = self.dim, hidden_size = 64, num_layers = 2, batch_first = True, bidirectional=True)
        self.en_lin = nn.Sequential(
            nn.Linear(64*2,self.latdim),
            nn.LeakyReLU()
        )
        self.de_LSTM = nn.LSTM(input_size = self.latdim, hidden_size = 64, num_layers = 2, batch_first = True, bidirectional=True)
        self.de_lins1 = nn.Linear(64*2, self.dim)
        self.de_lins2 = nn.Linear(64*2, self.dim)
        self.de_lins3 = nn.Linear(64*2, self.dim)

    def anomaly_score(self, x_hats, x):
        # Define anomaly score metric
        # as MSE between raw vs predicted.
        # or you can define own anomaly score inside forward step. e.g., rapp.
        low_std = torch.abs(x_hats[1]-x_hats[0])
        high_std = torch.abs(x_hats[2]-x_hats[1])
        absolute_err = torch.abs(x-x_hats[1])
        # score = F.mse_loss(x_hat,x,reduction='none')
        score = low_std + high_std + absolute_err
        if len(x.shape)==3:
            return score.mean(axis=(1,2)).detach()
        elif len(x.shape)==2:
            return score.mean(axis=(1)).detach()
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
        x_hat_q1 = self.de_lins1(x_hat1)
        x_hat_q2 = self.de_lins2(x_hat1)
        x_hat_q3 = self.de_lins3(x_hat1)
        
        # define anomaly score as mse between raw vs predicted.
        AnomalyScore = self.anomaly_score([x_hat_q1, x_hat_q2, x_hat_q3], x)
        return [x_hat_q1.detach(), x_hat_q2.detach(), x_hat_q3.detach()], AnomalyScore.detach()

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
        x_hat_q1 = self.de_lins1(x_hat1)
        x_hat_q2 = self.de_lins2(x_hat1)
        x_hat_q3 = self.de_lins3(x_hat1)

        # loss = F.mse_loss(x_hat, x)
        loss = self.Qloss([x_hat_q1,x_hat_q2,x_hat_q3], x)
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
        x_hat_q2 = self.de_lins2(x_hat1)
        loss = F.mse_loss(x_hat_q2, x)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # Autoencoder
        # x = x.view(x.size(0), -1)
        # embedding = self.encoder(x)
        # x_hat = self.decoder(embedding)

        # LSTM Autoencoder
        # z1,_ = self.en_LSTM(x)
        # embedding = self.en_lin(z1)
        # x_hat1,_ = self.de_LSTM(embedding)
        # x_hat_q2 = self.de_lins2(x_hat1)

        # loss = F.mse_loss(x_hat_q2, x)
        # self.log('test_loss', loss, on_step=True)

        z1,_ = self.en_LSTM(x)
        embedding = self.en_lin(z1)
        x_hat1,_ = self.de_LSTM(embedding)
        x_hat_q1 = self.de_lins1(x_hat1)
        x_hat_q2 = self.de_lins2(x_hat1)
        x_hat_q3 = self.de_lins3(x_hat1)
        
        # define anomaly score as mse between raw vs predicted.
        AnomalyScore = self.anomaly_score([x_hat_q1, x_hat_q2, x_hat_q3], x)
        return [x_hat_q1.detach(), x_hat_q2.detach(), x_hat_q3.detach()], AnomalyScore.detach()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor":"valid_loss"}
        #without lr scheduler           
        # return optimizer
