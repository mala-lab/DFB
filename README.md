# Improving Out-of-Distribution Detection with Disentangled Foreground and Background Features
By Choubo Ding, Guansong Pang

Official PyTorch implementation of the paper “Improving Out-of-Distribution Detection with Disentangled Foreground and Background Features”. The paper is available at [arXiv](https://arxiv.org/abs/2303.08727).

Code is modified from [Google BiT](https://github.com/google-research/big_transfer) and
[MOS](https://github.com/deeplearning-wisc/large_scale_ood).

## Usage

### 1. Dataset Preparation

#### In-distribution dataset

Please download CIFAR10 and CIFAR100, then place them in
`./dataset/cifar10` and  `./dataset/cifar100`, respectively.

#### Out-of-distribution dataset

We have curated 3 more OOD datasets from SVHN, Places, Textures.


1. **SVHN**:
   - Download from the [official website](http://ufldl.stanford.edu/housenumbers/).

2. **Places**:
   - Download from [MOS](https://github.com/deeplearning-wisc/large_scale_ood) or using the following command:
     ```bash
     wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
     ```

3. **Textures**:
   - Download from the [official website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

After downloading, please extract all datasets and place them in the `./dataset/` directory. 

### 2. Generating Pseudo Masks

In this step, we use a pre-trained classification model to generate pseudo masks. These masks will be used in the subsequent training process. Use the following command to generate pseudo masks:

```bash
python make_cam.py --dataset=../dataset/cifar10/train --model_path=cls_pretrained_models/cifar10.pth.tar --head_size=10
python make_cam.py --dataset=../dataset/cifar100/train --model_path=cls_pretrained_models/cifar100.pth.tar --head_size=100
```
- `--dataset`: Specifies the path to the training dataset
- `--model_path`: Specifies the path to the pre-trained model
- `--head_size`: Sets the size of the classification head (number of classes)

### 3. Training

To train the model, use the following command:

```bash
bash scripts/train.sh
```
This script will initiate the training process using the prepared datasets and generated pseudo masks.

### 4. Evaluation
After training, you can evaluate the model's performance using the following command:
```bash
bash scripts/eval.sh
```
This script will run the evaluation process on the test set and output the results.

Note: Make sure you have completed all previous steps (dataset preparation, pseudo mask generation, and training) before running the evaluation script.

## Citation

If you use our codebase, please cite our work:
```
@inproceedings{
ding2024improving,
title={Improving Out-of-Distribution Detection with Disentangled Foreground and Background Features},
author={Choubo Ding and Guansong Pang},
booktitle={ACM Multimedia 2024},
year={2024}
}
```
