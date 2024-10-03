import argparse

from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import os
from torch.utils.data import Subset

from model import resnetv2
import torchvision as tv

cudnn.enabled = True

import torch
from torchvision import datasets
import cam_to_mask


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        scales = (1.0, 0.5, 1.5, 2.0)
        x, y = super(ImageFolderWithPaths, self).__getitem__(index)
        ms_img_list = []
        for s in scales:
            if s == 1:
                s_img = x.numpy()
            else:
                s_img = pil_rescale(x, s, order=3)
            f_s = np.flip(s_img, -1)
            ms_img_list.append(np.stack([s_img, f_s], axis=0))
        if len(scales) == 1:
            ms_img_list = ms_img_list[0]
        return ms_img_list, y, self.imgs[index][0]


def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = "bicubic"
    elif order == 0:
        resample = 'nearest'

    img = torch.unsqueeze(img, 0)
    img = F.interpolate(img, size=size, mode=resample)

    return torch.squeeze(img, 0).numpy()

def pil_rescale(img, scale, order):
    height, width = img.shape[1:]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)


def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride

def split_dataset(dataset, n_splits):

    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=8 // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            print(iter)

            img_b, label, img_name = pack
            label = label[0]
            img_name = img_name[0]
            size = (128, 128)

            strided_size = get_strided_size(size, 4)
            strided_up_size = get_strided_up_size(size, 16)

            outputs = [model.forward_cam(img[0].cuda(non_blocking=True))
                       for img in img_b]
            if args.labeled:
                valid_cat = label.view(1, ).numpy()
            else:
                valid_cat = torch.argmax(outputs[0][1], dim=0).view(1, ).data.cpu().numpy()

            strided_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o, _ in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size, mode='bilinear', align_corners=False) for o, _ in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            root = os.path.dirname(img_name)
            img_name = os.path.basename(img_name).split('.')[0]
            class_dir = root.split('/')[-1]

            root = os.path.dirname(root) + '_cam'

            if not os.path.exists(os.path.join(root, class_dir)):
                os.makedirs(os.path.join(root, class_dir))
            np.save(os.path.join(root, class_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu().numpy(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = resnetv2.KNOWN_MODELS['BiT-M-R50x1'](head_size=args.head_size)
    state_dict = torch.load(args.model_path)
    model.load_state_dict_custom(state_dict['model'])
    model.eval()

    n_gpus = torch.cuda.device_count()

    crop = 128

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = ImageFolderWithPaths(args.dataset, val_tx)
    dataset = split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)

    # Dataset
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--labeled", type=bool, default=False)
    parser.add_argument("--head_size", default=100, type=int)

    args = parser.parse_args()
    run(args)
    cam_to_mask.run(args)