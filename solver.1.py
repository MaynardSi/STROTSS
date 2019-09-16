from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision as tv
import torchvision.transforms as transforms

from PIL import Image, ImageFile
import time
from tqdm import tqdm

from imageio import imread, imwrite

from loss_net import *
from transformation_net import *
# 


def pairwise_distances_sq_l2(x, y):
    # N x D
    x_norm = (x**2).sum(1).view(-1, 1)
    # D x N
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 1e-5, 1e5)/x.size(1)


def pairwise_distances_cos(x, y):
    x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y**2).sum(1).view(1, -1))

    dist = 1.-torch.mm(x, y_t)/x_norm/y_norm
    return dist

def load_path_for_pytorch(path, max_side=1000, force_scale=False):

    x = imread(path)                                                # Read image
    
    s = x.shape                                                     # Get image shape

    x = x/255.-0.5                                                  # Normalize Image and contrain to -0.5 to 0.5

    x = x.astype(np.float32)                                        # Type Cast to float

    x = torch.from_numpy(x).permute(2,0,1).contiguous()             # Permute from Ndarray (H X W X C) to Pytorch (C X H X W)

    ### If input image is taller than the size of the size we want, scale the input image to a given scale factor###
    if (max(s[:2])>max_side and max_side>0) or force_scale:    
        scale_factor = float(max_side)/max(s[:2])                   # Set scale factor to the longer dimension (Height or Width)

        x = F.interpolate(
            x.unsqueeze(0),
            (int(s[0] * scale_factor), int(s[1] * scale_factor)),
            mode='bilinear', align_corners=True)[0]

    return x

def compute_remd_loss(X, Y, loss="cosine"):
    # X e.g. pastiche features N x D points 
    # Y e.g. style features N x D points
    # assume already sampled

    # cost matrix N x N
    if loss == "cosine":
        cost_matrix = pairwise_distances_cos(X, Y)
    elif loss == "euclidean":
        cost_matrix = pairwise_distances_sq_l2(X, Y)

    m1, m1_inds = cost_matrix.min(dim = 1)
    m2, m2_inds = cost_matrix.min(dim = 0)

    # relaxed earth mover distance
    remd = torch.max(m1.mean(),m2.mean())

    return remd

def compute_moment_loss(X, Y):
    # X e.g. pastiche features N x D points 
    # Y e.g. style features N x D points
    # assume already sampled

    # 1 x D
    mu_x = X.mean(dim=0, keepdim=True)
    mu_y = Y.mean(dim=0, keepdim=True)

    dist_mu = torch.mean(torch.abs(mu_x - mu_y))

    # D x D
    cov_x = torch.mm((X-mu_x).transpose(0,1), (X-mu_x)) / X.size(0)
    cov_y = torch.mm((Y-mu_y).transpose(0,1), (Y-mu_y)) / Y.size(0)

    dist_cov = torch.mean(torch.abs(cov_x - cov_y))

    return dist_mu + dist_cov

def rgb2yuv(rgb):
    # rgb N x D

    # C 3 x 3
    C = torch.from_numpy(np.float32([[0.577350,0.577350,0.577350],[-0.577350,0.788675,-0.211325],[-0.577350,-0.211325,0.788675]]))
    
    if torch.cuda.is_available():
        C = C.cuda()

    # 3 x N
    yuv  = torch.mm(C,rgb.transpose(0,1))
    
    # N x 3
    yuv = yuv.transpose(0,1)
    return yuv

def compute_style_loss(pastiche_features, style_features, moment_weight = 1, content_weight = 1):
    # X e.g. pastiche features N x D points 
    # Y e.g. style features N x D points
    # composed of :
    #   - style_pastiche cosine remd
    #   - moment loss (mu and cov)
    #   - pixel color loss is euclidean remd between style image and pastiche image in YUV space

    loss_style_remd = compute_remd_loss(pastiche_features, style_features, loss="cosine")
    loss_moment = compute_moment_loss(pastiche_features, style_features)

    # extract image first 3 features is RGB
    pastiche_image = pastiche_features[:, :3]
    style_image = style_features[:, :3]

    pastiche_image_yuv = rgb2yuv(pastiche_image)
    style_image_yuv = rgb2yuv(style_image)

    loss_pixel_color = compute_remd_loss(pastiche_image_yuv, style_image_yuv, loss="euclidean")

    content_weight_frac = 1.0 / max(content_weight, 1.0)
    total_style_loss = loss_style_remd + moment_weight * (loss_moment + content_weight_frac * loss_pixel_color)

    return total_style_loss

def compute_content_loss(pastiche_features, content_features):
    # X e.g. pastiche features N x D points 
    # Y e.g. content features N x D points

    # D_pastiche - pairwise cosine distance with itself (N x N)
    D_pastiche = pairwise_distances_cos(pastiche_features, pastiche_features)

    # D_content - pairwise cosine distance with itself (N x N)
    D_content = pairwise_distances_cos(content_features, content_features)

    D_pastiche_norm = D_pastiche / D_pastiche.sum(dim=0, keepdim=True)
    D_content_norm = D_content / D_content.sum(dim=0, keepdim=True)

    content_loss = torch.mean(torch.abs(D_pastiche_norm - D_content_norm)) * pastiche_features.size(0)

    return content_loss

def image2lap_pyramid(X,levs, requires_grad=True, detach=True):
    pyr = []
    cur = X
    for i in range(levs):
        cur_x = cur.size(2)
        cur_y = cur.size(3)

        x_small = F.interpolate(cur, (max(cur_x//2,1), max(cur_y//2,1)), mode='bilinear', align_corners = True)
        x_back  = F.interpolate(x_small, (cur_x,cur_y), mode='bilinear', align_corners = True )
        lap = cur - x_back
        pyr.append(lap)
        cur = x_small

    pyr.append(cur)

    if detach:
        pyr = [x.detach().requires_grad_(requires_grad) for x in pyr]
    else:
        pyr = [x.requires_grad_(requires_grad) for x in pyr]

    return pyr

def lap_pyramid2image(pyr):

    cur = pyr[-1]
    levs = len(pyr)
    for i in range(0,levs-1)[::-1]:
        up_x = pyr[i].size(2)
        up_y = pyr[i].size(3)
        cur = pyr[i] + F.interpolate(cur,(up_x,up_y), mode='bilinear', align_corners = True)

    return cur


def style_transfer(pastiche_image, content_image, style_path, moment_weight, content_weight, long_side, lr=2e-3, num_iterations=250):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phi = LossNetwork()
    phi = phi.to(device)

    lap_pyramid = image2lap_pyramid(pastiche_image, 5, requires_grad=True, detach=True)

    optimizer =  optim.RMSprop(lap_pyramid, lr=lr)
    
    n_samples = 5000

    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        random_seed = np.random.randint(0,10000)
        pastiche_image = lap_pyramid2image(lap_pyramid)
        ### Extract content Features ###
        with torch.no_grad():
            content_features = phi.features_subsample(content_image, n_samples=n_samples, random_seed=random_seed, detach=True)

            ### Extract style featues ###
            style_image = load_path_for_pytorch(style_path, max_side = long_side, force_scale=True).unsqueeze(0).to(device)
            style_features = phi.features_subsample(style_image, n_samples=n_samples, random_seed=random_seed, detach=True)
        
        
        pastiche_features = phi.features_subsample(pastiche_image, n_samples=n_samples, random_seed=random_seed, detach=False)

        pastiche_features = pastiche_features.squeeze().transpose(0, 1)
        style_features = style_features.squeeze().transpose(0, 1)
        content_features = content_features.squeeze().transpose(0, 1)
        

        loss_content = compute_content_loss(pastiche_features, content_features)

        loss_style = compute_style_loss(pastiche_features, style_features, moment_weight = moment_weight, content_weight=content_weight)

        total_loss = (content_weight * loss_content + loss_style) / (1 + moment_weight + content_weight) # 1 because style weight is 1
        
        total_loss.backward()

        optimizer.step()

        if i % 10 ==0:
            print("Total Loss", total_loss.item())
            print("Style Loss" ,loss_style.item())
            print("Content Loss", loss_content.item())

    pastiche_image = lap_pyramid2image(lap_pyramid)

    return pastiche_image