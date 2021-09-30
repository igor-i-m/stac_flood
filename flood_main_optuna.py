import torch
import torch.nn as nn
import os
import yaml
import neptune.new as neptune
import pandas as pd
import sys
import segmentation_models_pytorch as smp
import numpy as np
import optuna
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, GaussNoise, RandomSizedCrop
from pytorch_toolbelt.optimization import GradualWarmupScheduler
from pytorch_toolbelt.inference import tta
from pytorch_toolbelt.losses import BinaryLovaszLoss, BinaryFocalLoss, JointLoss, DiceLoss
from torch.utils.data import DataLoader
from apex import amp
from tqdm import tqdm
from collections import namedtuple
from modules import STACDataset, Identity, seed_everything
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR


def smooth_label(target, alpha=0.2):
    target = (1-alpha)*target + alpha/2
    return target


def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=p),
        VerticalFlip(p=p),
        RandomRotate90(p=p),
        #RandomSizedCrop(min_max_height=[256,256], height=512, width=512, p=p)
        #GaussNoise(var_limit=0.1,p=p)
    ])


def IandU(pred, targ, threshold=0.5):
    pred_bin = pred>threshold
    targ_bin = targ>threshold
    
    intersection = torch.sum((pred_bin*targ_bin)>0).item()
    union = torch.sum((pred_bin+targ_bin)>0).item()
    
    return intersection, union


def setup(encoder_name, decoder_name, lr, args):
    
    if args.supplementary_data:
        in_channels=9
    else:
        in_channels=2
    model = getattr(smp, decoder_name)(encoder_name, in_channels=in_channels, activation=None).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )
    return model, optimizer

    
def train_epoch(trial, args, model, optimizer, dataloader, loss_fn, fold, lr_scheduler = None, run=None):
    losses = []
    model.train()
    running_loss = 0
    for step, (input_, target_)  in tqdm(enumerate(dataloader), total=len(dataloader),ncols=100, leave=False, desc='TRAINING'):
        optimizer.zero_grad()
        input_ = input_.cuda()
        target_ = target_.cuda()
        
        mask_ = 1.*(target_!=255)
        out_ = model(input_)
        out_ = out_*mask_
        if args.smooth_label:
            target_ = smooth_label(target_, alpha=args.labels_smooth_alpha)
        target_ = target_*mask_
        
        loss = loss_fn(out_, target_)
        losses.append(loss.item())
        running_loss += losses[-1]
        if (step+1)%args.loss_step == 0:
            if run:
                run[f'fold_{fold}/train/loss'].log(running_loss/args.loss_step)
            running_loss = 0
            
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
    if run:
        run[f'fold_{fold}/train/loss_std'].log(np.std(losses))
    
    return model
    

def validate(trial, args, model, dataloader, loss_fn, run, fold, data='val'):
    val_losses = []
    val_intersections = []
    val_unions = []
    model.eval()
    running_loss = 0
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for step, (input_, target_)  in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100, leave=False, desc='VALIDATION'):
            input_ = input_.cuda()
            target_ = target_.cuda()
            mask_ = 1.*(target_!=255)
            
            if args.tta:
                out_ = tta.d4_image2mask(model, input_)
            else:
                out_ = model(input_)
                
            out_ = out_*mask_
            target_ = target_*mask_
                
            out_ = sigmoid(out_) 
            loss = loss_fn(out_, target_)
            val_losses.append(loss.item())
            
            i_, u_ = IandU(out_, target_)
            val_intersections.append(i_)
            val_unions.append(u_)
            running_loss+=val_losses[-1]
            if (step+1)%args.loss_step == 0:
                if run and data=='val':
                    run[f'fold_{fold}/val/loss'].log(running_loss)
                running_loss = 0
                
    if run and data=='val':
        run[f'fold_{fold}/val/loss_std'].log(np.std(val_losses))
        
    aggregated_metric = np.sum(val_intersections)/(np.sum(val_unions)+1e-8)
    
    return np.mean(val_losses), aggregated_metric
    

def main(trial, params):
    
    if params['track']:
        run = neptune.init(
            project=params['neptune_project'],
            api_token=params['neptune_token'],
        )
        run['sys/tags'].add(params['run_tags']+[f'trial_{trial.number}'])
    else:
        run = None
        
    args = namedtuple('args', params.keys())(**params)
    
    print('STARTED TRIAL ', trial.number)
    
    seed_everything(args.seed)
    
    criterion = JointLoss(DiceLoss(mode='binary'), BinaryLovaszLoss())
    
    transforms=get_aug(p=0.5)
    
    best_val_metrics = []
    
    if args.save_ckpt:
        ckpt_path = os.path.join(args.ckpt_path, run['sys/id'].fetch())
        os.makedirs(ckpt_path, exist_ok=True)
    
    batch_size = trial.suggest_int('batch_size', 8,16, step=8)
    
    lr = trial.suggest_float('lr', args.low_lr, args.high_lr, log=True)
    
    encoder_name = trial.suggest_categorical('encoder', ['efficientnet-b2', 'resnet34','resnext50_32x4d'])
    decoder_name = trial.suggest_categorical('decoder', ['MAnet', 'Linknet', 'Unet', 'UnetPlusPlus'])
    
    run[f'parameters'] = {**params, **trial.params} 
    
    for fold_ in range(args.nfolds):
        #print('STARTED FOLD:', fold_)
        
        dataset_train = STACDataset(
            data_path=args.data_path,
            labels_path=args.labels_path,
            supplementary=args.supplementary_data,
            nfolds=args.nfolds,
            fold=fold_,
            train=True,
            tfms=transforms
        )
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=args.workers)
        
        dataset_train_val = STACDataset(
            data_path=args.data_path,
            labels_path=args.labels_path,
            supplementary=args.supplementary_data,
            nfolds=args.nfolds,
            fold=fold_,
            train=True
        )
        dataloader_train_val = DataLoader(dataset_train_val, batch_size=args.val_bs, shuffle=False, drop_last=False , num_workers=args.workers)
        
        dataset_val = STACDataset(
            data_path=args.data_path,
            labels_path=args.labels_path,
            supplementary=args.supplementary_data,
            nfolds=args.nfolds,
            fold=fold_,
            train=False
        )
        dataloader_val = DataLoader(dataset_val, batch_size=args.val_bs, shuffle=False, drop_last=False, num_workers=args.workers)
        
        model, optimizer = setup(encoder_name, decoder_name, lr, args)
        
        if args.apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, verbosity=0)
        
        if args.lr_scheduler:
            #lr_scheduler = OneCycleLR(
            #    optimizer,
            #    max_lr=args.lr,
            #    steps_per_epoch = len(dataloader_train),
            #    epochs=args.epochs
            #)
            lr_scheduler = CyclicLR(
                optimizer,
                base_lr=1.0e-7,
                max_lr=args.lr,
                step_size_up=250,
                mode='triangular',
                cycle_momentum=False
            )
            #lr_scheduler = GradualWarmupScheduler(
            #    optimizer=optimizer,
            #    multiplier=1.0,
            #    total_epoch=args.epochs
            #)
            
        else:
            lr_scheduler = None
            
        best_val_metric = 0
        
        for epoch in tqdm(range(args.epochs)):
        
            model = train_epoch(
                trial=trial,
                args=args,
                model=model,
                dataloader=dataloader_train,
                optimizer=optimizer,
                loss_fn=criterion,
                fold=fold_,
                lr_scheduler=lr_scheduler,
                run=run
            )
            
            if epoch % args.train_val_step == 0:
                fold_train_loss, fold_train_metric = validate(
                    trial=trial,
                    args=args,
                    model=model,
                    dataloader=dataloader_train_val,
                    loss_fn=criterion,
                    fold=fold_,
                    run=run,
                    data='train'
                )
                if run:
                    run[f'fold_{fold_}/train/IoU'].log(fold_train_metric)
            
            fold_val_loss, fold_val_metric = validate(
                trial=trial,
                args=args,
                model=model,
                dataloader=dataloader_val,
                loss_fn=criterion,
                fold=fold_,
                run=run,
                data='val'
            )
            
            if fold_val_metric > best_val_metric:
                best_val_metric = fold_val_metric
                
                ckpt = {
                    'model': model.state_dict()
                }
                
                path = os.path.join(
                    ckpt_path,
                    f'{encoder_name}.{decoder_name}.fold{fold_}.pth'
                )
                if os.path.isfile(path):
                    os.remove(path)
                    
                torch.save(ckpt, path)
            
            if run:
                run[f'fold_{fold_}/val/IoU'].log(fold_val_metric)
        
        best_val_metrics.append(best_val_metric)
        
        #print(f'Fold {fold_}: train loss {fold_train_loss}')
        #print(f'Fold {fold_}: train IoU {fold_train_metric}')
        
        print(f'Fold {fold_}: val loss {fold_val_loss}')
        print(f'Fold {fold_}: best val IoU {best_val_metric}')
                
        del dataset_train, dataloader_train, dataset_val, dataloader_val, dataset_train_val, dataloader_train_val
        del model, optimizer
        
        torch.cuda.empty_cache()

    run[f'mean_val_IoU'] = np.mean(best_val_metrics)
    
    run.stop()
    
    return np.mean(best_val_metrics)


if __name__=='__main__':
    
    with open('flood_config_optuna.yaml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params['gpu'])
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: main(trial, params), n_trials=params['n_trials'])#, timeout=600)
    
    best_trial = study.best_trial
    
    run['best_trial_params'] = best_trial.params
    print('BEST VALUE:', best_trial.value)