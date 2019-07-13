# https://www.kaggle.com/kenseitrg/simple-fastai-exercise
# https://www.kaggle.com/c/aerial-cactus-identification/discussion/90288#latest-571169
# https://www.kaggle.com/kernels/svzip/16989374

import os
import torch
import numpy as np
import pandas as pd
from fastai import metrics as fam
from matplotlib import pyplot as plt
from sklearn import metrics as slm

images_paths = ['./images/basic/', './images/losses/',
                './images/best_answers/', './images/model_comparisons/']
for images_path in images_paths:
    if not os.path.exists(images_path):
        os.mkdir(images_path)


def paint(scores: list, xlables: list, fname: str, bar_pads: list = ['.', '*', 'o'], colors: list = ['#60acfc', '#ff7c7c', '#feb64d']):
    # import matplotlib
    # %matplotlib notebook
    fig = plt.figure(figsize=(6, 6), dpi=72, facecolor="white")
    axes = plt.subplot(111)
    for i, score in enumerate(scores):
        axes.bar(0.5 + i, score, hatch=bar_pads[i],
                 color='white', edgecolor=colors[i], width=0.6, )
    axes.set_xlim(0, len(scores))
    axes.set_ylim(0, 1)
    axes.set_xticks(np.arange(0.5, len(scores)+0.5))
    axes.set_xticklabels(xlables)
    plt.savefig(fname)
    plt.close()
    # plt.show()


def get_scores(model_result_path: str, standard_answer: torch.Tensor):
    scores = dict()
    model_name = model_result_path.split('/')[-1]
    sub_len = len('submission_')  # + len(model_name) + 1
    best_score = 0
    best_answer = None
    best_loss_path = None
    for filename in os.listdir(model_result_path):
        path = os.path.join(model_result_path, filename)
        if not os.path.isfile(path):
            continue
        if filename.startswith('submission_'):
            filename = filename[sub_len: -4]
            answer = pd.read_csv(path).to_numpy()[:, 1].astype(np.float)
            score = fam.r2_score(
                standard_answer, torch.Tensor(answer)).item()
            scores[filename] = score
            if score > best_score:
                best_score = score
                best_answer = answer
                best_loss_path = path.replace('submission', 'loss')
    return model_name, scores, best_score, best_answer, best_loss_path


model_layer_dict = {
    'densenet': ['121', '161', '169'],
    'vgg': ['13_bn', '16_bn', '19_bn'],
    'resnet': ['50', '101', '152'],
    'dpn': ['68b', '92', '107'],
    'se_resnet': ['50', '101', '152'],
    'resnext': ['50', '101'],
}
lr_list = ['1e-1', '1e-2', '1e-3']  # 学习率
optim_list = ['adam', 'sgd', 'rprop']  # 优化器
one_cycle_list = ['use-one-cycle', 'not-use-one-cycle']  # 1 cycle 策略
transform_list = ['dofilp-flipvert', 'dofilp', 'nothing']  # 不同的数据增强方法对比


def find_score(key_name: str, scores: dict, default_score: float):
    for full_name, score in scores.items():
        if full_name.endswith(key_name):
            return score
    return default_score


def paint_wallper(scores: dict, keys: list, fname: str, default_score: float, bar_pads: list = ['.', '*', 'o']):
    section_scores = []
    for key in keys:
        section_scores.append(find_score(key, scores, default_score))
    print(keys, section_scores)
    paint(section_scores, keys, fname, bar_pads)


def paint_curve(ys: list, labels: list, title: str, fname: str, colors: list = ['#60acfc', '#ff7c7c', '#feb64d'], lw: int = 3):
    for i, y in enumerate(ys):
        label = labels[i] if labels is not None else None
        plt.plot(np.array(range(len(y))), np.array(
            y), label=label, color=colors[i % len(colors)], lw=lw)
    plt.axis()
    plt.title(title)
    plt.legend()
    plt.savefig(fname)
    plt.close()


def paint_scatter_and_bar(y: np.ndarray, colors: list, titles: list, fnames: list, marker='.'):
    # 散点图部分
    plt.scatter(np.arange(y.size), y, c=colors[0], marker=marker)
    plt.title(titles[0])
    plt.savefig(fnames[0])
    plt.close()
    # 柱状图部分
    n = 5
    heights = np.zeros(n)
    temp = [(i / n, (i + 1) / n) for i in range(n)]
    for value in y:
        for index, (start, end) in enumerate(temp):
            if start <= value < end:
                heights[index] += 1
                break
    plt.bar(np.arange(n) + 0.5, heights, 1, color=colors[1])
    plt.title(titles[1])
    plt.xticks(np.arange(n + 1), [i / n for i in range(n + 1)])
    plt.savefig(fnames[1])
    plt.close()


def generate_images(model_name: str, scores: dict, best_loss: np.ndarray, best_answer: np.ndarray, select_color: str):
    # 默认的
    default_score = find_score('flipvert', scores, None)
    if default_score is None:
        raise ValueError('scores should exists')
    print('default score', default_score)
    # 层数
    paint_wallper(scores, model_layer_dict[model_name], '{}{} layers comparison.jpg'.format(
        images_paths[0], model_name), default_score)
    # 学习率
    paint_wallper(scores, lr_list, '{}{} learning rate comparison.jpg'.format(
        images_paths[0], model_name), default_score)
    # 优化器
    paint_wallper(scores, optim_list, '{}{} optimizer comparison.jpg'.format(
        images_paths[0], model_name), default_score)
    # 1 cycle 策略
    paint_wallper(scores, one_cycle_list, '{}{} scheduler comparison.jpg'.format(
        images_paths[0], model_name), default_score, bar_pads=['.', '*'])
    # 数据增强
    paint_wallper(scores, transform_list, '{}{} data enhancement comparison.jpg'.format(
        images_paths[0], model_name), default_score)
    # best_loss
    paint_curve([best_loss], None, '{} loss curve'.format(model_name),
                '{}{} loss curve.jpg'.format(images_paths[1], model_name),
                colors=['#60acfc'])
    # 最好结果的散点图 + 对应的柱状图
    paint_scatter_and_bar(best_answer, [select_color, select_color],
                          ['{} answers\' scatter'.format(
                              model_name), '{} answers\' distribution'.format(model_name)],
                          ['{}{} answers\' scatter'.format(images_paths[2], model_name),
                           '{}{} answers\' distribution'.format(images_paths[2], model_name)])
    return default_score


def paint_wallper2(all_results: dict, default_scores: dict, keys: list, models: list, fname: str, title: str, colors: list = ['#60acfc', '#ff7c7c', '#feb64d'], markers: list = ['o-', '+-', '^-']):
    xs = np.arange(len(models))
    for i, key in enumerate(keys):
        section_scores = []
        for model_name in models:
            section_scores.append(find_score('{}_{}'.format(
                model_name, key), all_results, default_scores[model_name]))
        plt.plot(xs, section_scores, markers[i], color=colors[
            i], label=key)
    plt.axis()
    plt.title(title)
    # plt.ylim(0.6, 1)
    plt.xticks(xs, models)
    plt.legend()
    plt.savefig(fname)
    plt.close()


def paint_model_comparison(all_results: dict, default_scores: dict, models: list):
    # 学习率-模型对比图
    paint_wallper2(all_results, default_scores, lr_list, models,
                   '{}learning rate comparison.jpg'.format(images_paths[3]), 'learning rate comparison')
    # 优化器-模型对比图
    paint_wallper2(all_results, default_scores, optim_list, models,
                   '{}optimizer comparison.jpg'.format(images_paths[3]), 'optimizer comparison')
    # one cycle-模型对比图
    paint_wallper2(all_results, default_scores, one_cycle_list, models,
                   '{}scheduler comparison.jpg'.format(images_paths[3]), 'scheduler comparison')
    # 数据增强-模型对比图
    paint_wallper2(all_results, default_scores, transform_list, models,
                   '{}data enhancement comparison.jpg'.format(images_paths[3]), 'data enhancement comparison')


if __name__ == "__main__":
    model_result_paths = [
        './output/vgg', './output/densenet', './output/resnet', './output/se_resnet', './output/resnext']
    standard_answer = torch.Tensor(pd.read_csv(
        './good_submission.csv').to_numpy()[:, 1].astype(np.float))
    all_results = dict()
    default_scores = dict()
    best_losses = dict()
    colors = ['#60acfc', '#5bc49f', '#ff7c7c', '#feb64d', '#9287e7']
    for i, model_result_path in enumerate(model_result_paths):
        model_name, scores, best_score, best_answer, best_loss_path = get_scores(
            model_result_path, standard_answer)
        print(model_name, best_score, best_loss_path)
        for name, score in scores.items():
            print('   ', name, ":", score)
            all_results[name] = score
        best_losses[model_name] = pd.read_csv(best_loss_path).to_numpy()[
            :, 1].astype(np.float)
        default_scores[model_name] = generate_images(
            model_name, scores, best_losses[model_name], best_answer, colors[i])
    models = list(default_scores.keys())
    # 模型对比图
    paint_model_comparison(all_results, default_scores, models)
    # 总loss图 + 总acc图
    paint_curve(list(best_losses.values()), models, 'all models\' loss curve',
                '{}all models\' loss curve.jpg'.format(images_paths[3]), colors, 1)
