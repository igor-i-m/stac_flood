import torch
import pandas as pd
import glob
import pydicom
import sys
import numpy as np
import os
import random as rnd
import rasterio
from sklearn.model_selection import GroupKFold, StratifiedKFold
from .utils import sorted_nicely
from torch.utils.data import Dataset


def read_img_and_mask(fp):
    with rasterio.open(fp) as f:
        masked_arr = f.read(1, masked=True)
        data = masked_arr.data
        mask = masked_arr.mask
    return data, mask


def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,-1)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def get_classes_split(data_path, ids, bins):
    split = {}
    for id_ in ids:
        img, mask = read_img_and_mask(os.path.join(data_path, id_+'.tif'))
        img = img*(~mask)
        occupation = np.mean(img)
        class_ = np.digitize(occupation, bins, right=False)
        split[id_] = class_
    
    return split


class STACDataset(Dataset):
    def __init__(self, data_path, labels_path, nfolds, fold, bin_borders, supplementary=False, train=True, tfms=None):
        #kf = GroupKFold(n_splits=nfolds)
        kf = StratifiedKFold(n_splits=nfolds)
        ids = [fname[:5] for fname in os.listdir(labels_path)]
        #groups = [fname[:3] for fname in os.listdir(labels_path)]
        split = get_classes_split(labels_path, ids, bin_borders)
        self.fold_ids = [list(split.keys())[i] for i in list(kf.split(list(split.keys()), list(split.values())))[fold][0 if train else 1]]
        self.labels_path = labels_path
        self.data_path = data_path
        #self.fold_ids = [ids[i] for i in list(kf.split(ids, groups=groups))[fold][0 if train else 1]]
        self.train = train
        self.tfms = tfms
        self.supplementary = supplementary
        
    def __len__(self):
        return len(self.fold_ids)
    
    def __getitem__(self, idx):
        
        out = {}
        sample_id = self.fold_ids[idx]
        
        fp_vv = os.path.join(self.data_path,sample_id+'_vv.tif')
        img_vv, mask_vv = read_img_and_mask(fp_vv)
        
        fp_vh = os.path.join(self.data_path,sample_id+'_vh.tif')
        img_vh, mask_vh = read_img_and_mask(fp_vh)
        
        fp_target = os.path.join(self.labels_path, sample_id+'.tif')
        target, mask_target = read_img_and_mask(fp_target)
    
        img = np.stack([img_vv, img_vh], axis=-1)
        
        #norm
        min_norm = -77
        max_norm = 26
        img = np.clip(img, min_norm, max_norm)
        img = (img - min_norm) / (max_norm - min_norm)
        #
        #product = (img_vv**2)*(img_vh**2)
        #product = np.ma.masked_array(product, mask_vv | mask_vh)
        #min_product = product.min()
        #product = (product - min_product) / (product.max() - min_product)
        #product = np.expand_dims(product, -1)
        #img = np.concatenate([img, product.data], axis=-1)
        
        #img = np.expand_dims(product, -1)
        
        if self.supplementary:
            supplementary = filter(
                lambda path: ('vv' not in path) and ('vh' not in path),# and ('nasadem' not in path) and ('change' not in path),
                glob.glob(os.path.join(self.data_path, sample_id+'*.tif'))
            )
            
            sup_imgs = []
            for path in supplementary:
                img_, mask_ = read_img_and_mask(path)
                sup_imgs.append(img_)
            
            sup_imgs = np.stack(sup_imgs, axis=-1)
            sup_imgs = sup_imgs / 255.0
            
            img = np.concatenate([img, sup_imgs.astype('float32')], axis=-1)
        
        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=target)
            img, target = augmented['image'], augmented['mask']
        
        product_mask = target == 255
        product = (img[:,:,0]**2)*(img[:,:,1]**2)
        product = np.ma.masked_array(product, product_mask)
        min_product = product.min()
        product = (product - min_product) / (product.max() - min_product)
        #product = np.expand_dims(product, -1)
        #img = np.concatenate([img, product.data], axis=-1)
        img = np.insert(img, 2, product, axis=-1)
        
        out['input'] = img2tensor(img)
        out['target'] = img2tensor(target)
        
        #return img2tensor((img/255.0 - mean)/std),img2tensor(target)
        return out
        
        
        
        
        
        
        
        