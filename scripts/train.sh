#!/usr/bin/env bash

file_name=DFB_cifar10
MODEL=BiT-M-R50x1

python train.py \
--name exp1_${MODEL} \
--model ${MODEL} \
--bit_pretrained_dir bit_pretrained_models \
--logdir checkpoints/$file_name \
--eval_every 1000 \
--datadir ../dataset/cifar/cifar10

file_name=DFB_cifar100
MODEL=BiT-M-R50x1

python train.py \
--name exp1_${MODEL} \
--model ${MODEL} \
--bit_pretrained_dir bit_pretrained_models \
--logdir checkpoints/$file_name \
--eval_every 1000 \
--datadir ../dataset/cifar/cifar100
