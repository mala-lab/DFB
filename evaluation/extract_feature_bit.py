#!/usr/bin/env python
import argparse
import torch
import numpy as np
import pickle
from tqdm import tqdm
import mmcv
from os.path import dirname
import torchvision as tv
from model import resnetv2


def load_model(model_path, args):
    model = resnetv2.KNOWN_MODELS[args.model](head_size=args.logit_size, half_stride=args.half_stride)
    state_dict = torch.load(model_path)
    model.load_state_dict_custom(state_dict['model'])
    model.cuda().eval()
    return model


def extract_fc(model, fc_file):
    mmcv.mkdir_or_exist(dirname(fc_file))
    w = model.head.conv.weight.cpu().detach().squeeze().numpy()
    b = model.head.conv.bias.cpu().detach().squeeze().numpy()
    with open(fc_file, 'wb') as f:
        pickle.dump([w, b], f)
    return

def extract_feature(model, dataroot, out_file, args):
    torch.backends.cudnn.benchmark = True

    transform = tv.transforms.Compose([
        tv.transforms.Resize((128, 128)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    import os
    if dataroot.split('/')[-2] == 'SVHN':
        dataset = tv.datasets.SVHN(root=dataroot, download=True, transform=transform, split='test')
    else:
        dataset = tv.datasets.ImageFolder(dataroot, transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    features = []
    labels = []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.cuda()
            feat_batch = model(x, layer_index=5).cpu().numpy()
            features.append(feat_batch)
            labels.append(y.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    mmcv.mkdir_or_exist(dirname(out_file))
    with open(out_file, 'wb') as f:
        pickle.dump(features, f)

    if dataroot.split('/')[-1] == 'train':
        out_file = out_file.replace('.pkl', '_label.pkl')
        mmcv.mkdir_or_exist(dirname(out_file))
        with open(out_file, 'wb') as f:
            pickle.dump(features, f)