# In[0]
# !pip install pretrainedmodels
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastai import metrics
from fastai.vision import (DatasetType, ImageList, Learner, Path,
                           cnn_learner, error_rate, get_transforms,
                           imagenet_stats, accuracy,)

from densenet import densenet121, densenet161, densenet169
from resnet import (resnet50, resnet101, resnet152, resnext50_32x4d,
                    resnext101_32x8d)
from vgg import vgg13_bn, vgg16_bn, vgg19_bn
from se_resnet import se_resnet50, se_resnet101, se_resnet152


# In[1]
print(os.listdir("./input/"))
train_dir = "./input/train/train"
test_dir = "./input/test/train"
train = pd.read_csv('./input/train.csv')
test = pd.read_csv("./input/sample_submission.csv")
path = Path("./input")
device = torch.device('cuda:0')
test_img = ImageList.from_df(test, path=path/'test', folder='test')


# In[2]
def predict(learn: Learner, name: str):
    # submission.csv
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    test['has_cactus'] = preds.numpy()[:, 0]
    test.to_csv('submission_{}.csv'.format(name), index=False)
    print('Finish creating submission_{}.csv'.format(name))
    # loss.csv
    id_ = range(len(learn.recorder.losses))
    loss_df = pd.DataFrame(
        {'id': id_, 'loss': np.array(learn.recorder.losses)})
    loss_df.to_csv('loss_{}.csv'.format(name), index=False)
    print('Finish creating loss_{}.csv'.format(name))
    # 计算训练集上的一些指标
    preds, targets = learn.get_preds(ds_type=DatasetType.Train)
    preds_label = np.argmax(preds.numpy(), axis=1)
    id_ = range(len(preds))
    train_pred_df = pd.DataFrame({'id': id_, 'preds': preds.numpy(
    )[:, 0], 'preds_label': preds_label, 'targets': targets.numpy()})
    train_pred_df.to_csv('./train_pred_{}.csv'.format(name))
    print('Finish creating train_pred_{}.csv'.format(name))
    correct_count = np.equal(preds_label, targets.numpy()).sum()
    len_preds = len(preds)
    incorrect_count = len_preds - correct_count
    fpr, tpr = metrics.roc_curve(preds[:, 0], targets)
    fpr, tpr = fpr.numpy(), tpr.numpy()
    FP = np.floor(fpr * len_preds)
    FN = incorrect_count - FP
    TP = np.floor(tpr * len_preds)
    TN = correct_count - TP
    id_ = range(len(fpr))
    train_index_df = pd.DataFrame(
        {'id': id_, 'fpr': fpr, 'tpr': tpr,
         'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN})
    train_index_df.to_csv('./train_index_{}.csv'.format(name))
    print('Finish creating train_index_{}.csv'.format(name))
    # 销毁learn和保存模型
    learn.export('./model_{}.pth'.format(name), destroy=True)


model_normal_dict = {
    'densenet': densenet161,
    'vgg': vgg16_bn,
    'resnet': resnet101,
    'se_resnet': se_resnet101,
}
# 模型层数对比
model_layer_dict = {
    'densenet': {
        '121': densenet121,
        '161': None,
        '169': densenet169,
    },
    'vgg': {
        '13_bn': vgg13_bn,
        '16_bn': None,
        '19_bn': vgg19_bn,
    },
    'resnet': {
        '50': resnet50,
        '101': None,
        '152': resnet152,
    },
    'se_resnet': {
        '50': se_resnet50,
        '101': None,
        '152': se_resnet152,
    }
}
# 模型学习率对比
lr_dict = {
    '1e-1': 1e-1,
    # '1e-2': 1e-2,
    '1e-3': 1e-3,
}
# 优化器对比
optim_dict = {
    'adam': torch.optim.Adam,
    # 'sgd': torch.optim.SGD,
    'rprop': torch.optim.Rprop,
}
# 1 cycle 策略
one_cycle_dict = {
    # 'use-one-cycle': True,
    'not-use-one-cycle': False,
}
# 不同的数据增强方法对比
transform_dict = {
    # 'dofilp-flipvert': [True, True],
    'dofilp': [True, True],
    'nothing': [True, True],
}


# In[3]
for name, model in model_normal_dict.items():
    # 默认设置
    trfm = get_transforms(do_flip=True, flip_vert=True)
    train_img = (
        ImageList.from_df(train, path=path/'train', folder='train')
        .split_by_rand_pct(0.01)
        .label_from_df()
        .add_test(test_img)
        .transform(trfm, size=128)
        .databunch(path='.', bs=64, device=device)
        .normalize(imagenet_stats)
    )
    learn = cnn_learner(train_img, model, metrics=[
        error_rate, accuracy], opt_func=torch.optim.SGD)
    lr = 1e-2
    learn.fit_one_cycle(2, slice(lr))
    mld = model_layer_dict[name]
    predict(learn, '{}_{}_{}_{}_{}_{}'.format(name, list(mld.keys(
    ))[1], '1e-2', 'sgd', 'use-one-cycle', 'dofilp-flipvert'))
    # 层数
    for layer_name, true_model in mld.items():
        if true_model is None:
            continue
        learn = cnn_learner(train_img, true_model, metrics=[
            error_rate, accuracy], opt_func=torch.optim.SGD)
        lr = 1e-2
        learn.fit_one_cycle(2, slice(lr))
        predict(learn, '{}_{}'.format(name, layer_name))
    # 学习率
    for lr_name, lr in lr_dict.items():
        learn = cnn_learner(train_img, model, metrics=[
            error_rate, accuracy], opt_func=torch.optim.SGD)
        learn.fit_one_cycle(2, slice(lr))
        predict(learn, '{}_{}'.format(name, lr_name))
    # 优化器
    for optim_name, optim in optim_dict.items():
        learn = cnn_learner(train_img, model, metrics=[
            error_rate, accuracy], opt_func=optim)
        lr = 1e-2
        learn.fit_one_cycle(2, slice(lr))
        predict(learn, '{}_{}'.format(name, optim_name))
    # one cycle
    for cycle_name, cycle_strategy in one_cycle_dict.items():
        learn = cnn_learner(train_img, model, metrics=[
            error_rate, accuracy], opt_func=torch.optim.SGD)
        lr = 1e-2
        if cycle_strategy:
            learn.fit_one_cycle(2, slice(lr))
        else:
            learn.fit(2, slice(lr))
        predict(learn, '{}_{}'.format(name, cycle_name))
    # 数据增强策略
    for transform_name, transform_strategy in transform_dict.items():
        trfm = get_transforms(
            do_flip=transform_strategy[0],
            flip_vert=transform_strategy[1]
        )
        train_img = (
            ImageList.from_df(train, path=path/'train', folder='train')
            .split_by_rand_pct(0.01)
            .label_from_df()
            .add_test(test_img)
            .transform(trfm, size=128)
            .databunch(path='.', bs=64, device=device)
            .normalize(imagenet_stats)
        )
        learn = cnn_learner(train_img, model, metrics=[
            error_rate, accuracy], opt_func=torch.optim.SGD)
        lr = 1e-2
        learn.fit_one_cycle(2, slice(lr))
        predict(learn, '{}_{}'.format(name, transform_name))
