#!/usr/bin/env python3
# %%
import psutil
import torch
import torch.utils.data as data
from datetime import datetime
from dataset import dataset
from prefetcher import data_prefetcher
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


H, W = 256, 256

# %%
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nproc', type=int, default=psutil.cpu_count(logical=True))
    parser.add_argument('--gpu', type=int, default=torch.cuda.is_available())
    parser.add_argument('--file_list')
    parser.add_argument('--savename', default='res')
    parser.add_argument('--model', default='/data/20201018/model/model-2020-10-18-20-14-46-086057-best.pth')
    parser.add_argument('--bs', type=int, default=1)

    args = parser.parse_args()
    return args


# %%
# Test the trained model
def predict(args):
    with open(args.file_list, 'r') as f:
        files = f.read().splitlines()
    ds = dataset(files)
    data_loader = data.DataLoader(
        ds, batch_size=args.bs,
        sampler=data.SequentialSampler(ds),
        num_workers=args.nproc, pin_memory=args.gpu, drop_last=False)
    print('[%s] Start test using: %s.' % (datetime.now(), args.model.split('/')[-1]))

    # %%
    # Test the trained model
    print('[%s] Start test.' % datetime.now())
    if args.gpu:
        net = torch.load(args.model)
    else:
        net = torch.load(args.model, map_location=torch.device('cpu'))
    print('[%s] Model loaded: %s' % (datetime.now(), args.model))

    # start test
    net.eval()
    correct, total = 0, 0
    pred = torch.empty((len(ds), H, W), dtype=torch.long)
    labels = torch.empty((len(ds), H, W), dtype=torch.long)
    if args.gpu:
        pred = pred.cuda()
        labels = labels.cuda()
    prefetcher = data_prefetcher(data_loader, args.gpu)
    with torch.no_grad():
        inputs, targets, index = prefetcher.next()
        k = 0
        while (inputs is not None) and (targets is not None):
            if args.bs == 1:
                sampleID = files[index].split('/')[-1].split('.')[0]
            outputs = net(inputs)  # with shape NCHW
            _, predict = torch.max(outputs.data, 1)  # with shape NHW
            correct_i = (predict == targets).sum().item()
            total += targets.shape[0]
            correct += correct_i
            if args.bs == 1:
                print('[%5d/%5d]    %s    test_accu: %.3f' % (total, len(files), sampleID, correct_i/H/W))
            else:
                print('[%5d/%5d] test_accu: %.3f' % (total, len(files), correct_i/H/W/targets.shape[0]))

            pred[index.tolist()] = predict
            labels[index.tolist()] = targets

            # prefetch train data
            inputs, targets, index = prefetcher.next()
            k += 1
    print('Average test_accu: %.3f' % (correct/total/H/W))
    print('[%s] Finished test.' % datetime.now())

    return pred, labels


def get_cm(y_pred, y_true, n_classes):
    y_pred = torch.flatten(y_pred)
    y_true = torch.flatten(y_true)
    indices = n_classes * y_true + y_pred
    cm = torch.bincount(indices, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    return cm


def cm_metric(cm, eps):
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN+eps)
    # Specificity or true negative rate
    TNR = TN/(TN+FP+eps)
    # Precision or positive predictive value
    PPV = TP/(TP+FP+eps)
    # Negative predictive value
    NPV = TN/(TN+FN+eps)
    # Fall out or false positive rate
    FPR = FP/(FP+TN+eps)
    # False negative rate
    FNR = FN/(TP+FN+eps)
    # False discovery rate
    FDR = FP/(TP+FP+eps)

    precision = TP/(TP+FP+eps)
    recall = TP/(TP+FN+eps)
    specificity = TN/(TN+FP+eps)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    F1 = 2*TP/(2*TP+FP+FN+eps)
    FWIoU = (TP+FN)/(TP+FP+TN+FN)*TP/(TP+FP+FN+eps)

    metric = {'tp': TP, 'tn': TN, 'fp': FP, 'fn': FN, 'prec': precision, 'recall': recall, 'spec': specificity, 'accu': accuracy, 'f1': F1, 'fwiou': FWIoU}

    return metric


def plotCM(labels, matrix, gt, t, savename, annotation, cmap, cbarlabel, threshold=None, textcolors=("black", "white")):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap, vmin=0.0, vmax=1.0)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # set the origin at bottom left
    # ax.invert_yaxis()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(matrix.max())/2.

    # Loop over data dimensions and create text annotations.
    if annotation == True:
        for i in range(len(labels)):
            if gt[i, i] != 0:
                text = ax.text(i, i, '%.2f' % matrix[i, i], ha='center', va='center', color=textcolors[int(im.norm(matrix[i, i]) < threshold)], fontsize=5)
            for j in range(len(labels)):
                if (i != j) and (matrix[i, j] >= t) and (gt[i, j] != 0):
                    text = ax.text(j, i, '%.2f' % matrix[i, j], ha='center', va='center', color=textcolors[int(im.norm(matrix[i, j]) < threshold)], fontsize=5)

    # cbar = fig.colorbar(im, label='Recall')#, ticks=list(range(count+1)))
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # ax.set_title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.savefig(savename+'.png', bbox_inches='tight', dpi=600)


def fwiou(cm):
    freq = np.sum(cm, axis=1) / np.sum(cm)
    iu = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    n_classes = 6
    args = get_args()
    labels = list(range(n_classes))

    y_pred, y_true = predict(args)
    # print(y_pred.shape)
    # print(y_true.shape)
    torch.save(y_pred, args.savename+'_y_pred.pt')
    torch.save(y_true, args.savename+'_y_true.pt')

    # if args.gpu:
    #     y_pred = torch.load("y_pred32.pt")
    #     y_true = torch.load("y_true32.pt")
    # else:
    #     y_pred = torch.load("y_pred32.pt", map_location=torch.device('cpu'))
    #     y_true = torch.load("y_true32.pt", map_location=torch.device('cpu'))
    cm = get_cm(y_pred, y_true, n_classes)
    cm = cm.cpu().numpy()[1:, 1:]
    # np.save("cm.npy", cm)
    # sys.exit(0)

    eps = 1e-8
    # cm = np.load('cm_train.npy')
    TOTAL = cm.sum()*np.ones(cm.shape)
    FN = np.dot(cm.sum(1, keepdims=True), np.ones((1, cm.shape[1]))) - cm
    FP = np.dot(np.ones((cm.shape[0], 1)), cm.sum(0, keepdims=True)) - cm
    TN = TOTAL - FN - FP - cm

    precision = cm/(cm+FP+eps)
    recall = cm/(cm+FN+eps)
    specificity = TN/(TN+FP+eps)
    accuracy = (cm+TN)/TOTAL
    f1 = 2*cm/(2*cm+FP+FN+eps)
    iou = cm/(cm+FP+FN+eps)
    gt = cm+FN
    # annotate non-diag when value >= t
    # plotCM(labels[1:], recall, gt, t=0.1, savename=args.savename, annotation=True, cmap=None, cbarlabel='Recall')
    plotCM(labels[1:], recall, gt, t=0.1, savename=args.savename+'_recall.png', annotation=True, cmap=None, cbarlabel='Recall')
    plotCM(labels[1:], f1, gt, t=0.1, savename=args.savename+'_f1.png', annotation=True, cmap=None, cbarlabel='F1')
    plotCM(labels[1:], iou, gt, t=0.1, savename=args.savename+'_iou.png', annotation=True, cmap=None, cbarlabel='IoU')

    print('mRecall:', np.sum(np.diag(recall))/(n_classes-1))
    print('mSpecificity:', np.sum(np.diag(specificity))/(n_classes-1))
    print('mF1:', np.sum(np.diag(f1))/(n_classes-1))
    print('mIoU:', np.sum(np.diag(iou))/(n_classes-1))

    metric = cm_metric(cm, eps)
    print(metric)

    FWIoU = fwiou(cm)
    print(FWIoU)
