# In[0]
# !pip install pretrainedmodels
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretrainedmodels as ptms
import torch
import torch.nn as nn
import torchvision.models as tmodels
from fastai import metrics
from fastai.vision import (DatasetType, ImageList, Learner, Path,
                           cnn_learner, error_rate, get_transforms,
                           imagenet_stats, models, accuracy,)

# In[1]
print(os.listdir("./input/"))
print(dir(tmodels))
print(dir(models))

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
    # Calculate some metrics on the training set
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
    train_index_df = pd.DataFrame({
        'id': id_, 'fpr': fpr, 'tpr': tpr,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN})
    train_index_df.to_csv('./train_index_{}.csv'.format(name))
    print('Finish creating train_index_{}.csv'.format(name))
    # Destroy learn and save the model
    learn.export('./model_{}.pth'.format(name), destroy=True)


def get_model(model_name, pretrained, seq=False, pname='imagenet'):
    pretrained = pname if pretrained else None
    model = getattr(ptms, model_name)(pretrained=pretrained)
    return nn.Sequential(*model.children()) if seq else model


def se_resnet50(pretrained: bool = False): return get_model(
    'se_resnet50', pretrained)


def se_resnet101(pretrained: bool = False): return get_model(
    'se_resnet101', pretrained)


def se_resnet152(pretrained: bool = False): return get_model(
    'se_resnet152', pretrained)


# In[3]
model_normal_dict = {
    'densenet': tmodels.densenet161,
    'vgg': tmodels.vgg16_bn,
    'resnet': tmodels.resnet101,
    'resnext': tmodels.resnext101_32x8d,
    'se_resnet': se_resnet101,
}
# comparison of layer
model_layer_dict = {
    'densenet': {
        '121': tmodels.densenet121,
        # '161': tmodels.densenet161,
        '161': None,
        '169': tmodels.densenet169
    },
    'vgg': {
        '13_bn': tmodels.vgg13_bn,
        # '16_bn': tmodels.vgg16_bn,
        '16_bn': None,
        '19_bn': tmodels.vgg19_bn,
    },
    'resnet': {
        '50': tmodels.resnet50,
        # '101': tmodels.resnet101,
        '101': None,
        '152': tmodels.resnet152,
    },
    'se_resnet': {
        '50': se_resnet50,
        # '101': se_resnet101,
        '101': None,
        '152': se_resnet152,
    },
    'resnext': {
        '101': None,
        '50': tmodels.resnext50_32x4d,
    },
}
# Comparison of model learning rate
lr_dict = {
    '1e-1': 1e-1,
    # '1e-2': 1e-2,
    '1e-3': 1e-3,
}
# Comparison of model optimizer
optim_dict = {
    'adam': torch.optim.Adam,
    # 'sgd': torch.optim.SGD,
    'rprop': torch.optim.Rprop,
}
# 1 cycle or not
one_cycle_dict = {
    # 'use-one-cycle': True,
    'not-use-one-cycle': False,
}
# Pre-processing or not
transform_dict = {
    # 'dofilp-flipvert': [True, True],
    'dofilp': [True, True],
    'nothing': [True, True],
}


# In[4]
for name, model in model_normal_dict.items():
    # default setting
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
    # the number of layer
    for layer_name, true_model in mld.items():
        if true_model is None:
            continue
        learn = cnn_learner(train_img, true_model, metrics=[
            error_rate, accuracy], opt_func=torch.optim.SGD)
        lr = 1e-2
        learn.fit_one_cycle(2, slice(lr))
        predict(learn, '{}_{}'.format(name, layer_name))
    # learning rate
    for lr_name, lr in lr_dict.items():
        learn = cnn_learner(train_img, model, metrics=[
            error_rate, accuracy], opt_func=torch.optim.SGD)
        learn.fit_one_cycle(2, slice(lr))
        predict(learn, '{}_{}'.format(name, lr_name))
    # optimizer
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
    # pre-processing
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
