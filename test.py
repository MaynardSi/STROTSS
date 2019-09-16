from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as f

import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision as tv
import torchvision.transforms as transforms

from PIL import Image, ImageFile

# from tqdm import tqdm

from loss_net import *
from transformation_net import *


inp = torch.Tensor(3, 256, 256)


def tensor_normalizer(x):
    t = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    return t(x)
def tensor_denormalizer(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.255]).unsqueeze(-1).unsqueeze(-1)

    return x * std + mean
tensor_denormalizer(1)