import torch
import torch.nn as nn
import os
import yaml
import neptune.new as neptune
import pandas as pd
import sys
import segmentation_models_pytorch as smp
import numpy as np
import copy
import cv2
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, GaussNoise, RandomSizedCrop, ElasticTransform, ShiftScaleRotate, GaussianBlur
from pytorch_toolbelt.optimization import GradualWarmupScheduler
from pytorch_toolbelt.inference import tta
from pytorch_toolbelt.losses import BinaryLovaszLoss, BinaryFocalLoss, JointLoss, DiceLoss, SoftBCEWithLogitsLoss, BiTemperedLogisticLoss
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
        GaussianBlur(p=p),
        #ShiftScaleRotate(
        #                    shift_limit=0.,
        #                    scale_limit=0.,
        #                    rotate_limit=45,
        #                    p=p, 
        #                    border_mode=cv2.BORDER_REFLECT),
        #ElasticTransform(p=p, alpha=5)
    ])


def IandU(pred, targ, threshold=0.5):
    pred_bin = pred>threshold
    targ_bin = targ>threshold
    
    intersection = torch.sum((pred_bin*targ_bin)>0).item()
    union = torch.sum((pred_bin+targ_bin)>0).item()
    
    return intersection, union


def setup(args):
    if args.supplementary_data:
        in_channels = 10
    else:
        in_channels = 3
    model = getattr(smp, args.decoder)(args.encoder, in_channels=in_channels, activation=None).cuda()
    optimizer = torch.optim.Adam( 
        model.parameters(), lr=args.lr
    )
    return model, optimizer


def train_epoch(args, model, optimizer, dataloader, loss_fn, fold, aux_loss_fn=None, lr_scheduler = None, run=None):
    losses = []
    model.train()
    running_loss = 0
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100, leave=False, desc='TRAINING'):
        optimizer.zero_grad()
        input_ = batch['input'].cuda()
        target_ = batch['target'].cuda()
        
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
            
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
    if run:
        run[f'fold_{fold}/train/loss_std'].log(np.std(losses))
    
    return model


def validate(args, model, dataloader, loss_fn, run, fold, data='val'):
    val_losses = []
    val_intersections = []
    val_unions = []
    model.eval()
    running_loss = 0
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for step, batch  in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100, leave=False, desc='VALIDATION'):
            input_ = batch['input'].cuda()
            target_ = batch['target'].cuda()
            mask_ = 1.*(target_!=255)            
            
            if args.tta:
                out_ = tta.d4_image2mask(model, input_)
            else:
                out_ = model(input_)
                
            out_ = out_*mask_
            target_ = target_*mask_
            
            loss = loss_fn(out_, target_)
            val_losses.append(loss.item())
            out_ = sigmoid(out_)
            
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
    

def main(args, run=None):
    
    seed_everything(args.seed)
    
    print('BATCH SIZE:', args.train_bs)
    print('LEARNING RATE:', args.lr)
    
    criterion = JointLoss(DiceLoss(mode='binary'), BinaryLovaszLoss())
    #criterion = JointLoss(DiceLoss(mode='binary'), BiTemperedLogisticLoss(t1=0.5, t2=2.0))

    transforms=get_aug(p=0.5)
    
    if args.save_ckpt:
        ckpt_path = os.path.join(args.ckpt_path, run['sys/id'].fetch())
        os.makedirs(ckpt_path, exist_ok=True)
    
    best_val_metrics = []
    
    for fold_ in range(args.nfolds):
        
        print('STARTED FOLD:', fold_)
        
        dataset_train = STACDataset(
            data_path=args.data_path,
            labels_path=args.labels_path,
            supplementary=args.supplementary_data,
            nfolds=args.nfolds,
            fold=fold_,
            train=True,
            tfms=transforms,
            bin_borders = args.bin_borders
        )
        dataloader_train = DataLoader(dataset_train, batch_size=args.train_bs, shuffle=True, num_workers=args.workers)
        
        dataset_train_val = STACDataset(
            data_path=args.data_path,
            labels_path=args.labels_path,
            supplementary=args.supplementary_data,
            nfolds=args.nfolds,
            fold=fold_,
            train=True,
            bin_borders = args.bin_borders
        )
        dataloader_train_val = DataLoader(dataset_train_val, batch_size=args.val_bs, shuffle=False, drop_last=False , num_workers=args.workers)
        
        dataset_val = STACDataset(
            data_path=args.data_path,
            labels_path=args.labels_path,
            supplementary=args.supplementary_data,
            nfolds=args.nfolds,
            fold=fold_,
            train=False,
            bin_borders = args.bin_borders
        )
        dataloader_val = DataLoader(dataset_val, batch_size=args.val_bs, shuffle=False, drop_last=False, num_workers=args.workers)
        
        model, optimizer = setup(args)
        if args.lr_scheduler:
            lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=args.lr,
                steps_per_epoch = len(dataloader_train),
                epochs=args.epochs
            )
            #lr_scheduler = CyclicLR(
            #    optimizer,
            #    base_lr=1.0e-7,
            #    max_lr=args.lr,
            #    step_size_up=250,
            #    mode='triangular',
            #    cycle_momentum=False
            #)
            #lr_scheduler = GradualWarmupScheduler(
            #    optimizer=optimizer,
            #    multiplier=1.0,
            #    total_epoch=args.epochs
            #)
            
        else:
            lr_scheduler = None
        
        best_val_metric = 0
        best_ckpt_path = ''
        
        for epoch in tqdm(range(args.epochs)):
        
            model = train_epoch(
                args=args,
                model=model,
                dataloader=dataloader_train,
                optimizer=optimizer,
                loss_fn=criterion,
                fold=fold_,
                lr_scheduler=lr_scheduler,
                run=run
            )
            
            if epoch % args.train_val_step == 0 :
                fold_train_loss, fold_train_metric = validate(
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
                    f'{args.encoder}.{args.decoder}.fold{fold_}.best_{epoch}.pth'
                )
                
                if len(best_ckpt_path):
                    os.remove(best_ckpt_path)
                    
                torch.save(ckpt, path)
                
                best_ckpt_path = path
            
            if run:
                run[f'fold_{fold_}/val/IoU'].log(fold_val_metric)
                
            ckpt = {
                'model': model.state_dict()
            }
            
            path = os.path.join(
                ckpt_path,
                f'{args.encoder}.{args.decoder}.fold{fold_}.last_checkpoint.pth'
            )
            if os.path.isfile(path):
                os.remove(path)
            
            torch.save(ckpt, path)
                
        best_val_metrics.append(best_val_metric)
        
        #print(f'Fold {fold_}: train loss {fold_train_loss}')
        #print(f'Fold {fold_}: train IoU {fold_train_metric}')
        
        print(f'Fold {fold_}: val loss {fold_val_loss}')
        print(f'Fold {fold_}: best val IoU {best_val_metric}')
                
        del dataset_train, dataloader_train, dataset_val, dataloader_val, dataloader_train_val, dataset_train_val
        
    run['mean_val_IoU'] = np.mean(best_val_metrics)


if __name__=='__main__':
    
    with open('flood_config.yaml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        
    if params['track']:
        run = neptune.init(
            project=params['neptune_project'],
            api_token=params['neptune_token'],
        )
        run['parameters'] = params
        run['sys/tags'].add(params['run_tags'])
        
    args = namedtuple('args', params.keys())(**params)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    main(args, run if args.track else None)