from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch.optim as optim

import cv2
import os
import glob

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision as tv
import torchvision.transforms as transforms

import time
from imageio import imread, imwrite

from tqdm import tqdm, tnrange, tqdm_notebook

from loss_net import *
from solver import *

def main(content_path='input_img.jpg', style_path='style_img.jpg', max_scale=5, output_path='./output.png'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    small_side = 64
    content_weight = 16.0
    moment_weight = 1.0
    num_iterations = 250
    ### Run Style Transfer ###
    # scale = [1,2,3,4]
    st = time.time()
    for scale in range(1, max_scale):
        print("Scale =", scale)
        long_side = small_side * (2 ** (scale - 1))
        lr = 2e-3
        # def __init__(self, content_image, style_path, long_side):
        ### Loading Content Image ###
        content_image = load_path_for_pytorch(content_path,long_side,force_scale=True).unsqueeze(0).to(device)
        style_image = load_path_for_pytorch(style_path, long_side, force_scale=True).unsqueeze(0).to(device)
        style_image_mean = style_image.mean(2,keepdim=True).mean(3,keepdim=True)
        s = Solver(content_image, content_path, style_path, long_side, scale)

        ### Initialize bottom level of Laplacian pyramid for content image at the current scale ###
        laplacian = content_image.clone() - F.interpolate(
            F.interpolate(content_image, 
                (content_image.size(2)//2, content_image.size(3)//2),
                mode = 'bilinear', 
                align_corners = True),
            (content_image.size(2),
            content_image.size(3)),
            mode='bilinear', 
            align_corners = True)
        # scale = 1 first iteration
        
        if scale == 1:
            stylized_image = style_image_mean + laplacian
        # scale = 4 last iteration
        elif scale == max_scale - 1:
            stylized_image = F.interpolate(stylized_image.detach(),(content_image.size(2),content_image.size(3)), mode = 'bilinear', align_corners = True)
            lr = 1e-3
        # scale = [2,3] in-between iterations
        else:
            stylized_image = F.interpolate(stylized_image.detach
            (),(content_image.size(2),content_image.size(3)), mode = 'bilinear', align_corners = True) + laplacian
        
        ### Style Transfer at this scale ###
        stylized_image = s.style_transfer(stylized_image, moment_weight, content_weight, lr, num_iterations = num_iterations)

        ### Decrease Content Weight for next scale ###
        content_weight = content_weight/2.0
    print("End {}".format(time.time() - st))
    output_image = torch.clamp(stylized_image[0],-0.5,0.5).data.cpu().numpy().transpose(1,2,0)

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m%d%Y_%H%M%S", named_tuple)
    imwrite("{}{}_{}_{}.jpg".format(output_path, time_string, (os.path.basename(content_path)).replace(".jpg",""), (os.path.basename(style_path)).replace(".jpg","")), output_image)

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    print("asdasdas")
    for stylepath in glob.iglob('style_images/*'):
        for contentpath in glob.iglob('content_images/*'):
            print("asdasdas")
            print("Style: {} \t\t\tContent: {}" .format(stylepath, contentpath))
            main(content_path=contentpath, style_path=stylepath, max_scale=4, output_path="./output_images/")
        
    