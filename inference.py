from __future__ import division
import os
from PIL import Image
from skimage import transform as sktsf
from data.dataset import Dataset, TestDataset,inverse_normalize
from data.dataset import pytorch_normalze
import numpy as np
import pdb
import torch
import matplotlib
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.config import opt
from torch.autograd import Variable
from torch.utils import data as data_
from utils import array_tool as at
from data.util import  read_image
import pandas as pd
from PIL import Image
import attacks
import argparse

WIDER_BBOX_LABEL_NAMES = (
    'Face')

parser = argparse.ArgumentParser(description='FaceShield Inference')
parser.add_argument('--ep', type=int, default=1,
                    help='Epsilon')

def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
         (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray:
        A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    try:
        img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
    except:
        ipdb.set_trace()
    # both the longer and shorter should be less than
    # max_size and min_size
    normalize = pytorch_normalze
    return normalize(img)

def add_bbox(ax,bbox,label,score):
    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()
        label_names = list(WIDER_BBOX_LABEL_NAMES) + ['bg']
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax

if __name__ == '__main__':
    args = parser.parse_args()
    attacker = attacks.DCGAN(train_adv=False)
    attacker.load('min_max_attack.pth')
    img = read_image('stock_1.jpg')
    img = preprocess(img)
    img = torch.from_numpy(img)[None]
    im_path = 'stock_1.jpg'
    img = Variable(img.float().cuda())
    im_path_clone = b = '%s' % im_path
    ori_img_ = inverse_normalize(at.tonumpy(img[0]))
    adv_img = attacker.perturb(img,epsilon=args.ep)
    adv_img_ = inverse_normalize(at.tonumpy(adv_img[0]))
    ori_img_ = ori_img_.transpose((1,2,0))
    adv_img_ = adv_img_.transpose((1,2,0))

