import torch
import torch.nn as nn
import sys
import numpy as np
import os
import rasterio
import glob
from torch.utils.data import Dataset


def read_img_and_mask(fp):
    with rasterio.open(fp) as f:
        masked_arr = f.read(1, masked=True)
        data = masked_arr.data
        mask = masked_arr.mask
    return data, mask


def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,0)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class STACDataset(Dataset):
    def __init__(self, data_path, supplementary=False, tfms=None):
        ids = set()
        for name in os.listdir(os.path.join(data_path,'test_features')):
            ids.add(name[:5])
        self.ids = list(ids)
        self.data_path = data_path
        self.tfms = tfms
        self.supplementary = supplementary
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        
        fp_vv = os.path.join(self.data_path,'test_features',sample_id+'_vv.tif')
        img_vv, mask_vv = read_img_and_mask(fp_vv)
        
        fp_vh = os.path.join(self.data_path,'test_features',sample_id+'_vh.tif')
        img_vh, mask_vh = read_img_and_mask(fp_vh)
        
        img = np.stack([img_vv, img_vh], axis=-1)
        
        min_norm = -77
        max_norm = 26
        img = np.clip(img, min_norm, max_norm)
        img = (img - min_norm) / (max_norm - min_norm)
        
        product = (img[:,:,0]**2)*(img[:,:,1]**2)
        product = np.ma.masked_array(product, mask_vv | mask_vh)
        min_product = product.min()
        product = (product - min_product) / (product.max() - min_product)
        product = np.expand_dims(product, -1)
        img = np.concatenate([img, product.data], axis=-1)
        
        if self.supplementary:
            supplementary = filter(
                lambda path: ('vv' not in path) and ('vh' not in path),
                glob.glob(os.path.join(self.data_path, '*', sample_id+'.tif'))
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
            img, target = augmented['image'],augmented['mask']
            
        return img2tensor(img), sample_id
    
    
def harmonic_mean(inp):
    n = len(inp)
    inverted = inp.pow(-1.0)
    hm = n / torch.sum(inverted, dim=0)
    return hm
    
    
class Ensembler(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, inp):
        
        outputs = [model(inp) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(dim=0)
        #return harmonic_mean(outputs)
        