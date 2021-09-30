import os
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import glob
import ttach as tta
import sys
from modules import STACDataset, Ensembler
#from pytorch_toolbelt.inference import tta,# Ensembler
from torch.utils.data import DataLoader
from tifffile import imsave


def setup(encoder_name, decoder_name, supplementary=False):
    if supplementary:
        in_channels=9
    else:
        in_channels=3
    model = getattr(smp, decoder_name)(encoder_name, in_channels=in_channels, activation=None, encoder_weights=None).cuda()
    return model


def parse(path):
    name = path.split('/')[-1]
    encoder_name, decoder_name = name.split('.')[0:2]
    return encoder_name, decoder_name


def threshold(img, threshold=0.5):
    thresholded = (img >= threshold).astype('uint8')
    return thresholded


def create_model(model_paths, supplementary=False):
    
    models = []
    for path in model_paths:
        ckpt = torch.load(path)
        encoder_name, decoder_name = parse(path)
        model = setup(encoder_name, decoder_name, supplementary)
        model.load_state_dict(ckpt['model'])
        models.append(model)
    model = Ensembler(models)
    return model


def inference(model, dataloader, output_path):
    
    model.eval()
    sigmoid = nn.Sigmoid()
    
    with torch.no_grad():
        for inp, ids  in dataloader:
            inp = inp.cuda()
            out = model(inp)
            out = sigmoid(out)
            for img, id in zip(out, ids):
                output_f = os.path.join(output_path,f'{id}.tif')
                binary = threshold(img.detach().cpu().numpy(), 0.5)
                imsave(output_f, binary)
                

def main():
    
    supplementary_data = False
    
    model_paths = glob.glob('./checkpoints/*')
    model = create_model(model_paths, supplementary=supplementary_data)
    model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    model.cuda()
    
    data_path = './data/'
    
    dataset = STACDataset(
        data_path,
        tfms=None,
        supplementary=supplementary_data
    )    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )
    
    output_path = './submission/'
    
    inference(model, dataloader, output_path)


if __name__=='__main__':
    
    main()