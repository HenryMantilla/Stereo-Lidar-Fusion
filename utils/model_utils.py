import os

import torch
import torch.optim as optim

def save_checkpoint(ckpt_dir, model, optim, scheduler, epoch):

    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    states = {
        'model': model_state_dict,
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }

    ckpt_path = os.path.join(ckpt_dir, 'checkpoint_epoch_{:02d}.ckpt'.format(epoch))
    torch.save(states, ckpt_path)

    return ckpt_path


def load_checkpoint(model, optimizer, scheduler, ckpt_path, weights_only):

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, weights_only)
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if not weights_only:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch']

def get_optimizer(args, model):
    optimizer_classes = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'adamw': optim.AdamW
    }

    if args.optimizer not in optimizer_classes:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    optimizer_params = {
        'params': model.parameters(),
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }

    optimizer = optimizer_classes[args.optimizer](**optimizer_params)

    return optimizer

def get_lr_scheduler(args, optimizer):
    lr_schedulers = {
        'step': optim.lr_scheduler.StepLR,
        'cosine': optim.lr_scheduler.CosineAnnealingLR, #t_max, eta_min
        'cyclic': optim.lr_scheduler.CyclicLR, #base_lr, max_lr, step_size_up, mode
    }

    if args.scheduler not in lr_schedulers:
        raise ValueError(f"Unknown optimizer scheduler {args.scheduler}")

    lr_schedulers_params = {
        'step': {
            'step_size': 25,
            'gamma': 0.1
        },
        'cosine': {
            'T_max': args.epochs, 
            'eta_min': 1e-6,  
        },
        'cyclic': {
            'base_lr': 0.0001, 
            'max_lr': 0.001,  
            'step_size_up': 10,  
            'mode': 'triangular', 
        },
    }

    scheduler = lr_schedulers[args.scheduler]
    scheduler_params = lr_schedulers_params[args.scheduler]

    lr_scheduler = scheduler(optimizer, **scheduler_params)

    return lr_scheduler
