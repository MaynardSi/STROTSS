from __future__ import print_function
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models.vgg as vgg
import time

import numpy as np


###### MODEL FILE #######

LossOutput = namedtuple(
    "LossOutput", ["image", "relu1_1", "relu1_2", "relu2_1", "relu2_2", "relu3_1", "relu3_2", "relu3_3", "relu4_3", "relu5_3"
])

class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg16(pretrained=True).features
        # self.vgg_layers = vgg.vgg16(pretrained=True)
        self.layer_name_mapping = {
            '1' : "relu1_1", 
            '3' : "relu1_2", 
            '6' : "relu2_1", 
            '8' : "relu2_2", 
            '11' : "relu3_1", 
            '13' : "relu3_2", 
            '15' : "relu3_3", 
            '22' : "relu4_3", 
            '29' : "relu5_3", 
        }
        self.content_features_raw = None
        self.style_features_raw = None

    def forward(self, x):
        output = {}
        output["image"] = x
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
    
        return LossOutput(**output)

    def features_subsample(self, iteration, image, image_type, n_samples=1000, random_seed=0,  detach=False ):
        ###
        if image_type != 'pastiche':
            if image_type == 'content':
                if iteration == 0:
                    self.content_features_raw = self.forward(image)
                features_raw = self.content_features_raw
                  
            elif image_type == 'style':
                if iteration == 0:
                    self.style_features_raw = self.forward(image)  
                features_raw = self.style_features_raw
        else:
            features_raw = self.forward(image)
       
        # features_raw = self.forward(image)

        ### Create a mesh grid of H*W ###

        H = np.array(range(image.size(2)))          # Creates an array from 0 to H
        W = np.array(range(image.size(3)))          # Creates an array from 0 to W
        nx, ny = np.meshgrid(H, W)   
        
        nx = np.expand_dims(nx.flatten(),1)         # Create x points
        
        ny = np.expand_dims(ny.flatten(),1)         # Create y points  
        coordinates = np.concatenate([nx, ny], 1)   # Pair created x points and y points for coordinates
        n_coordinates = coordinates.shape[0]        # Count number of coordinates
        ### Create indices for coordinate random sample ###
        # sample_coordinate_index = np.random.randint(0, n_coordinates, size=(n_samples))
        # st_shuffle = time.time()
        # np.random.seed(random_seed)
        # np.random.shuffle(coordinates)
        # print("shuffle ", time.time() - st_shuffle)
        ### If tensor size is less than the specified number of samples, use all points intead ###
        n_sample_points = min(n_samples, n_coordinates)
        np.random.seed(random_seed)
        coordinates_idx= np.random.choice(np.arange(n_coordinates), n_sample_points)

        # Create an array of selected points based on the created index ###

        # xx = coordinates[:n_sample_points, 0]
        # yy = coordinates[:n_sample_points, 1]
        xx = coordinates[coordinates_idx, 0]
        yy = coordinates[coordinates_idx, 1]
        ### Selects the corresponding value of the sampled points in the image and puts them in the list ###
        if not detach :
            image = image.requires_grad_(True)

        # sampled_image = [image[:,:, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(n_sample_points)]

        sampled_image = image[:,:, xx, yy]
        ### To check
        # sampled_feature_list = torch.cat(sampled_image, 2)

        layer_sampled_features = [sampled_image]

        ### For each features extracted from network ###
        for i in range(len(features_raw)):

            layer_features = features_raw[i]  # store outputs of layer i
            
            ### Check if you have already downscaled, if not, then downscale ###
            if i>0 and features_raw[i].size(2) < features_raw[i-1].size(2):
                xx = xx/2.0
                yy = yy/2.0

            ### Clip values to 0 to max size and floor floating values (0.5 -> 0) ###
            xx = np.clip(xx,0,layer_features.size(2)-1).astype(np.int32) 
            yy = np.clip(yy,0,layer_features.size(3)-1).astype(np.int32)

            ### Selects the corresponding value of the sampled points in the feature map and puts them in the list ###
            # sampled_features = [layer_features[:,:, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(n_sample_points)]
            sampled_features = layer_features[:,:, xx, yy]

            ### Concatenate tensors along axis 2 (not sure why)
            # sampled_features_list = torch.cat(sampled_features,2)
            
            ### Append sampled points of feature map
            # layer_sampled_features.append(sampled_features_list)
            layer_sampled_features.append(sampled_features)

        # features_sampled = torch.cat([layer.contiguous() for layer in layer_sampled_features],1)

        features_sampled = torch.cat(layer_sampled_features,dim = 1)

        if detach:
            features_sampled = features_sampled.detach()

        return features_sampled, features_raw
