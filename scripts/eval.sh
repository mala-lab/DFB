#!/usr/bin/env bash

file_name=DFB_cifar10
MODEL=BiT-M-R50x1
exp_name=exp1_${MODEL}
python eval_cls.py \
--model ${MODEL} \
--id_train_path ../dataset/cifar10/train \
--id_test_path ../dataset/cifar10/test \
--model_path checkpoints/$file_name/$exp_name/bit.pth.tar \
--logdir checkpoints/$file_name \
--logit_size 100 \
--ood_path ../dataset/cifar100/test ../dataset/SVHN/test ../dataset/Places365/test ../dataset/Textures/test

file_name=DFB_cifar100
MODEL=BiT-M-R50x1
exp_name=exp1_${MODEL}
python eval_cls.py \
--model ${MODEL} \
--id_train_path ../dataset/cifar100/train \
--id_test_path ../dataset/cifar100/test \
--model_path checkpoints/$file_name/$exp_name/bit.pth.tar \
--logdir checkpoints/$file_name \
--logit_size 100 \
--ood_path ../dataset/cifar10/test ../dataset/SVHN/test ../dataset/Places365/test ../dataset/Textures/test


