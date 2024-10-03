import argparse
import os
import numpy as np
import imageio
from torch import multiprocessing
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import datasets
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Subset

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        x, y = super(ImageFolderWithPaths, self).__getitem__(index)

        return CHW_to_HWC(x), y, self.imgs[index][0]

def CHW_to_HWC(img):
    return np.transpose(img, (1, 2, 0))

def split_dataset(dataset, n_splits):

    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def _work(process_id, infer_dataset, args):

    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(infer_data_loader):
        _, _, img_name = pack
        img_name = img_name[0]


        root = os.path.dirname(img_name)
        img_name = os.path.basename(img_name).split('.')[0]
        class_dir = root.split('/')[-1]

        mask_root = os.path.dirname(root) + '_mask'
        root = os.path.dirname(root) + '_cam'

        if os.path.exists(os.path.join(mask_root, class_dir, img_name + '.png')):
            print('Exsising: ' + img_name)
            continue
        else:
            print('Creating: ' + img_name)

        cam_dict = np.load(os.path.join(root,class_dir, img_name + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']

        cams = CHW_to_HWC(cams)

        cams = gaussian_filter(cams, sigma=4)

        cams -= np.min(cams)

        if np.max(cams) != 0:
            cams /= np.max(cams)

        conf = cams > 0.5

        conf = conf * (cam_dict['keys'] + 1)

        if not os.path.exists(os.path.join(mask_root, class_dir)):
            os.makedirs(os.path.join(mask_root, class_dir))
        imageio.imwrite(os.path.join(mask_root,class_dir, img_name + '.png'), conf.astype(np.uint8))
        if process_id == args.num_workers - 1 and iter % (len(databin) // 20) == 0:
            print("%d " % ((5 * iter + 1) // (len(databin) // 20)))

def run(args):
    crop = 128

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = ImageFolderWithPaths(args.dataset, val_tx)
    dataset = split_dataset(dataset, args.num_workers)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
    print(']')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run(args)