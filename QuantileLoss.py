#%%
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantileLoss(nn.Module):
    def __init__(self, quantiles = [0.1,0.5,0.9], reduction='mean'):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        self.reduction = reduction
        self.qs = torch.Tensor(quantiles)

    def forward(self, y_preds, y_true):
        assert len(self.quantiles)==y_preds.size()[-1]
        assert not y_true.requires_grad       
        #comment out if your model contains a sigmoid or equivalent activation layer
        assert y_true.size() ==y_preds.size()[:,-1]
        ### N : batch, L : window, C : Channel, 
        ### y_preds == N x L x C x 
        y_trues = y_true.unsqueeze(-1).expand(*(y_true.size()),len(self.quantiles))
        errors = y_trues - y_preds
        if y_trues.device!=torch.device('cpu'):
            self.qs=self.qs.to(y_trues.device)
        losses = torch.max(
            errors*(self.qs-1), errors*self.qs
        )
        ret = torch.sum(losses,dim=-1)

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)

        return ret
