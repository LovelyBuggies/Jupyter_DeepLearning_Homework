## code1

code1主要由两部分组成。第一部分是 code1.py ，里面使用到了 pretrainedmodels 和 torchvision 这两个包含很多现成的预训练好的模型的 pytroch 模型库。第二部分是 requirements.txt ，``pip install -r requirements.txt`` 就可以下载所需的指定版本的库。

## code2

code2主要由两部分组成。第一部分是5个 python 文件，包含 resnet.py 、 densenet.py 、 se_resnet.py 、 vgg.py 、 code2.py，里面共有 5 种模型，其中 resnet 和 resnext 都在 resnet.py中，而其他模型如文件名所示。还有主文件 code2.py ，它从其余四个文件中导入所需模型。第二部分也是 requirements.txt ，但这一部分的 requirements.txt 不包含 pretrainedmodels 和 torchvision 。