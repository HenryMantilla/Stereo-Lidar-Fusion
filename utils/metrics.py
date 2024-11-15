import torch
import torch.nn as nn
from easydict import EasyDict

class MetricEvaluator():
    def __init__(self, metrics):

        self.metrics_fn = EasyDict()
        self.results = EasyDict()

        for metric in metrics:
            self.metrics_fn[metric] = globals()[metric]
            self.results[metric] = 0.

        self.metrics = metrics


    def evaluate_metrics(self, pred, groundtruth):
        
        valid_mask = (groundtruth > 0)
        pred_valid = pred[valid_mask]
        groundtruth_valid = groundtruth[valid_mask]

        for metric in self.metrics:
            self.results[metric] = self.metrics_fn[metric](pred_valid, groundtruth_valid)
        
        return EasyDict(self.results.copy())

## remember convert from disparity to depth before evaluate metrics
def mae_metric(pred, groundtruth):

    mae = torch.abs(pred - groundtruth).mean()
    return mae

def imae_metric(pred, groundtruth):

    imae = torch.abs(1000.*(1./pred - 1./groundtruth)).mean()
    return imae

def rmse_metric(pred, groundtruth):
    
    mse_loss = nn.MSELoss(reduction='mean')
    rmse = torch.sqrt(mse_loss(pred, groundtruth))
    return rmse

def irmse_metric(pred, groundtruth):

    mse_loss = nn.MSELoss(reduction='mean')
    irmse = torch.sqrt(mse_loss((1000.*(1./pred), 1000.*(1./groundtruth))))
    return irmse

### d1 is calculated on disparity
def d1_metric(pred, groundtruth):
    
    mae = torch.abs(pred-groundtruth)
    d1_mask = (mae > 3) & (mae / groundtruth.abs() > 0.05)

    return torch.mean(d1_mask.float())

