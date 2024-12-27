import torch
import torch.nn as nn
from easydict import EasyDict

class MetricEvaluator:
    def __init__(self, metrics):
        self.metrics_fn = EasyDict()
        self.results = EasyDict()
        for metric in metrics:
            self.metrics_fn[metric] = globals()[metric]
            self.results[metric] = 0.
        self.metrics = metrics

    def evaluate_metrics(self, pred, groundtruth):
        
        valid_mask = (groundtruth > 1e-8) & (groundtruth < 220.0)
             # Apply mask
        pred_valid = pred[valid_mask]
        groundtruth_valid = groundtruth[valid_mask]
        
        pred_mm = pred_valid * 1000.0
        groundtruth_mm = groundtruth_valid * 1000.0
        
        pred_km = pred_valid / 1000.0 
        groundtruth_km = groundtruth_valid / 1000.0 
        
        pred_inv_km = 1.0 / (pred_km + 1e-8)
        groundtruth_inv_km = 1.0 / (groundtruth_km + 1e-8)
        
        # Calculate metrics
        for metric in self.metrics:
            if metric in ['mae_metric', 'rmse_metric']:
                self.results[metric] = self.metrics_fn[metric](pred_mm, groundtruth_mm)
            elif metric in ['imae_metric', 'irmse_metric']:
                self.results[metric] = self.metrics_fn[metric](pred_inv_km, groundtruth_inv_km)
        
        return EasyDict(self.results.copy())

def mae_metric(pred_mm, groundtruth_mm):
    """Mean Absolute Error in mm"""
    return torch.abs(pred_mm - groundtruth_mm).mean()

def rmse_metric(pred_mm, groundtruth_mm):
    """Root Mean Square Error in mm"""
    return torch.sqrt(torch.pow(pred_mm - groundtruth_mm, 2).mean())

def imae_metric(pred_inv_km, groundtruth_inv_km):
    """Inverse Mean Absolute Error in 1/km"""
    return torch.abs(pred_inv_km - groundtruth_inv_km).mean()

def irmse_metric(pred_inv_km, groundtruth_inv_km):
    """Inverse Root Mean Square Error in 1/km"""
    return torch.sqrt(torch.pow(pred_inv_km - groundtruth_inv_km, 2).mean())

