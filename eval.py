#!/usr/bin/env python
import argparse
import os

import torch
import numpy as np
from tqdm import tqdm
import mmcv
from numpy.linalg import norm, pinv
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from os.path import basename, splitext
from scipy.special import logsumexp
import pandas as pd
from evaluation.extract_feature_bit import extract_fc, extract_feature, load_model
from evaluation.test_utils import make_output_folders, write_score_file, vis_score
from utils.test_utils import get_measures

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--logdir', help='Path to checkpoint')
    parser.add_argument('--model_path', help='Path to checkpoint')
    parser.add_argument('--id_train_path', help='Path to data')
    parser.add_argument('--id_test_path', help='Path to data')
    parser.add_argument('--ood_path', nargs='+', help='Path to ood dataset')
    parser.add_argument('--clip_quantile', default=0.99, help='Clip quantile to react')
    parser.add_argument('--batch', type=int, default=256, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Workers for dataloader')
    parser.add_argument('--prefix', default='', help='Name for different eval exp')
    parser.add_argument('--logit_size', type=int, default=100, help='Sets the size of the classification head')
    parser.add_argument('--model', default='BiT-M-R50x1', help='Bit model')
    parser.add_argument('--half_stride', type=bool, default=False)

    return parser.parse_args()

#region Helper
def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out

def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

#endregion

#region OOD

def gradnorm(x, w, b, logit_size):
    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.from_numpy(x).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []

    for i in tqdm(x):
        targets = torch.ones((1, logit_size)).cuda()
        fc.zero_grad()
        loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(fc.weight.grad.data)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)

#endregion

def main():
    args = parse_args()

    print(f"Loading model.")
    # model = None
    model = load_model(args.model_path, args)

    make_output_folders(args.logdir, args.prefix)

    id_name = args.id_train_path.split('/')[-2]
    ood_names = [ood.split('/')[-2] for ood in args.ood_path]
    print(f"ood datasets: {ood_names}")

    fc_file = "{}/tensors/{}.pkl".format(args.logdir, 'fc')

    if os.path.exists(fc_file) is not True:
        print(f"Extract fc file.")
        extract_fc(model, fc_file)
    w, b = mmcv.load(fc_file)
    print(f'{w.shape=}, {b.shape=}')

    recall = 0.95

    print('load features')
    id_train_file = "{}/tensors/{}_{}.pkl".format(args.logdir,  id_name, 'train')
    id_test_file = "{}/tensors/{}_{}.pkl".format(args.logdir, id_name, 'test')

    if os.path.exists(id_train_file) is not True:
        print(f"Extract {id_name} train features.")
        extract_feature(model, args.id_train_path, id_train_file, args)
    feature_id_train = mmcv.load(id_train_file).squeeze()
    id_label_file = id_train_file.replace('.pkl', '_label.pkl')
    train_labels = mmcv.load(id_label_file).squeeze()

    if os.path.exists(id_test_file) is not True:
        print(f"Extract {id_name} test features.")
        extract_feature(model, args.id_test_path, id_test_file, args)
    feature_id_test = mmcv.load(id_test_file).squeeze()

    feature_oods = {}
    for name, path in zip(ood_names, args.ood_path):
        ood_file = "{}/tensors/{}.pkl".format(args.logdir, name)
        if os.path.exists(ood_file) is not True:
            print(f"Extract {name} features.")
            extract_feature(model, path, ood_file, args)
        feature_oods[name] = mmcv.load(ood_file).squeeze()

    print(f'{feature_id_train.shape=}, {feature_id_test.shape=}')
    for name, ood in feature_oods.items():
        print(f'{name} {ood.shape}')

    print('computing logits...')

    logit_id_train = feature_id_train @ w.T + b
    logit_id_test = feature_id_test @ w.T + b
    logit_oods = {name: feat @ w.T + b for name, feat in feature_oods.items()}


    print('computing softmax...')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_test = softmax(logit_id_test, axis=-1)
    softmax_oods = {name: softmax(logit, axis=-1) for name, logit in logit_oods.items()}

    u = -np.matmul(pinv(w), b)

    df = pd.DataFrame(columns = ['method', 'oodset', 'auroc', 'fpr'])

    dfs = []

    # ---------------------------------------
    method = 'MSP'
    print(f'\n{method}')
    result = []
    score_id = softmax_id_test.max(axis=-1)
    write_score_file(args.logdir, args.prefix, method, id_name + "_id", score_id)
    for name, softmax_ood in softmax_oods.items():
        score_ood = softmax_ood.max(axis=-1)
        write_score_file(args.logdir, args.prefix, method, name, score_ood)
        auroc, aupr_in, aupr_out, fpr95 = get_measures(score_id.reshape((-1, 1)), score_ood.reshape((-1, 1)))
        result.append(dict(method=method, oodset=name, auroc=auroc, fpr=fpr95, aupr_in=aupr_in, aupr_out=aupr_out))
        print(f'{method}: {name} auroc {auroc:.4%}, fpr {fpr95:.4%}, aupr_in {aupr_in:.4%}, aupr_out {aupr_out:.4%}')
        vis_score(args.logdir, args.prefix, method, id_name, name, score_id, score_ood)

    df = pd.DataFrame(result)
    dfs.append(df)
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # # ---------------------------------------
    method = 'ODIN'
    T = 2
    print(f'\n{method}')
    result = []
    score_id = softmax(logit_id_test / T, axis=-1).max(axis=-1)
    write_score_file(args.logdir, args.prefix, method, id_name + "_id", score_id)
    for name, logit_ood in logit_oods.items():
        score_ood = softmax(logit_ood / T, axis=-1).max(axis=-1)
        write_score_file(args.logdir, args.prefix, method, name, score_ood)
        auroc, aupr_in, aupr_out, fpr95 = get_measures(score_id.reshape((-1, 1)), score_ood.reshape((-1, 1)))
        result.append(dict(method=method, oodset=name, auroc=auroc, fpr=fpr95, aupr_in=aupr_in, aupr_out=aupr_out))
        print(f'{method}: {name} auroc {auroc:.4%}, fpr {fpr95:.4%}, aupr_in {aupr_in:.4%}, aupr_out {aupr_out:.4%}')
        vis_score(args.logdir, args.prefix, method, id_name, name, score_id, score_ood)

    df = pd.DataFrame(result)
    dfs.append(df)
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # ---------------------------------------
    method = 'Energy'
    print(f'\n{method}')
    result = []
    score_id = logsumexp(logit_id_test, axis=-1)
    write_score_file(args.logdir, args.prefix, method, id_name + "_id", score_id)
    for name, logit_ood in logit_oods.items():
        score_ood = logsumexp(logit_ood, axis=-1)
        write_score_file(args.logdir, args.prefix, method, name, score_ood)
        auroc, aupr_in, aupr_out, fpr95 = get_measures(score_id.reshape((-1, 1)), score_ood.reshape((-1, 1)))
        result.append(dict(method=method, oodset=name, auroc=auroc, fpr=fpr95, aupr_in=aupr_in, aupr_out=aupr_out))
        print(f'{method}: {name} auroc {auroc:.4%}, fpr {fpr95:.4%}, aupr_in {aupr_in:.4%}, aupr_out {aupr_out:.4%}')
        vis_score(args.logdir, args.prefix, method, id_name, name, score_id, score_ood)

    df = pd.DataFrame(result)
    dfs.append(df)
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')

    # # ---------------------------------------
    method = 'ViM'
    print(f'\n{method}')
    result = []
    DIM = 1000 if feature_id_test.shape[-1] >= 2048 else 512
    print(f'{DIM=}')

    print('computing principal space...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    print('computing alpha...')
    vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f'{alpha=:.4f}')

    vlogit_id_val = norm(np.matmul(feature_id_test - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_test, axis=-1)
    score_id = -vlogit_id_val + energy_id_val
    write_score_file(args.logdir, args.prefix, method, id_name + "_id", score_id)
    for name, logit_ood, feature_ood in zip(ood_names, logit_oods.values(), feature_oods.values()):
        energy_ood = logsumexp(logit_ood, axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        score_ood = -vlogit_ood + energy_ood
        write_score_file(args.logdir, args.prefix, method, name, score_ood)
        auroc, aupr_in, aupr_out, fpr95 = get_measures(score_id.reshape((-1, 1)), score_ood.reshape((-1, 1)))
        result.append(dict(method=method, oodset=name, auroc=auroc, fpr=fpr95, aupr_in=aupr_in, aupr_out=aupr_out))
        print(f'{method}: {name} auroc {auroc:.4%}, fpr {fpr95:.4%}, aupr_in {aupr_in:.4%}, aupr_out {aupr_out:.4%}')
        vis_score(args.logdir, args.prefix, method, id_name, name, score_id, score_ood)

    df = pd.DataFrame(result)
    dfs.append(df)
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')


    dfs = pd.concat(dfs)
    csv_file = "{}/results_{}/result.csv".format(args.logdir, args.prefix)
    dfs.to_csv(csv_file)

if __name__ == '__main__':
    main()
