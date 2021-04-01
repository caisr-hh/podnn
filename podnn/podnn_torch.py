# Copyright 2021 The PODNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=====================================================================================

"""

This file provides five different classes for creating PODNN architectures in Torch.

The five modules are as follows:

- InputLayer: Prepares data in parallel form to be consumable by the upcoming layers


- ParallelLayer: creates a parallel sub-layer formed from unit it receives.


- OrthogonalLayer1D: makes the outputs of previous parallel sub-layer orthognal to each other.
                     The output of each unit of the previous sub-layer needs to be 1D vector.


- OrthogonalLayer2D: makes the filter outputs of previous parallel convolutional sub-layer orthognal to each other.
                     In other words, OrthogonalLayer2D is applied on each corresponding filters between models
                     within the ensemble.


- AggregationLayer:  aggregates the output of all individual models which makes it ready to be fed to the meta
                     part of the ensemble.

"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


n_models_global = 5
agg_out_dim = 3


class InputLayer(nn.Module):

    """
       InputLayer stucture the data in a parallel form ready to be consumed by
       the upcoming parallel layes.
    """


    def __init__(self,n_models):

        """
        Arg:
            n_models: number of individual models within the ensemble
        """

        super(InputLayer, self).__init__()
        self.n_models = n_models
        global n_models_global
        n_models_global = self.n_models

    def forward(self,x):

        """
        Arg
            x: is the input to the network as in other standard deep neural networks.

        return:
            x_parallel: is the parallel form of the received input (x).
        """

        x_parallel = torch.unsqueeze(x,0)
        x_parallel_next = torch.unsqueeze(x, 0)
        for i in range(1,self.n_models):
            x_parallel =  torch.cat((x_parallel,x_parallel_next),axis=0)

        return x_parallel


class ParallelLayer(nn.Module):

    """
        Parallellayer creates a parallel layer from the structure of unit_model it receives.
    """

    def __init__(self, unit_model):

        """
        Arg:
            unit_model: specifies what computational module each unit of the parallel layer contains.
                        unit_model is a number of layer definitions followed by each other.
        """

        super(ParallelLayer,self).__init__()
        self.n_models = n_models_global
        self.model_layers = []
        for i in range(self.n_models):
            for j in range(len(unit_model)):
                try:
                    unit_model[j].reset_parameters()
                except:
                    pass
            self.model_layers.append(deepcopy(unit_model))
        self.model_layers = nn.ModuleList(self.model_layers)

    def forward(self, x):

        """
        Arg:
            x: is the parallel input with shape [n_models,n_samples,dim] for fully connected layers and
                                                [n_samples,n_models,n_channels,im_hight,im_width] for convlutional layers.

        return:
            parallel_output: is the output formed by applying modules within each units on the input.

            shape: [n_models,n_samples,dim] for fully connected layers and
                   [n_samples,n_models,n_channels,im_hight,im_width,] for convlutional layers.

        """

        parallel_output = self.model_layers[0](x[0])
        parallel_output = torch.unsqueeze(parallel_output,0)
        for i in range(1,self.n_models):
            next_layer = self.model_layers[i](x[i])
            next_layer = torch.unsqueeze(next_layer, 0)
            parallel_output = torch.cat((parallel_output,next_layer),0)
            parallel_output = torch.cat((parallel_output,next_layer),0)
        return parallel_output


class OrthogonalLayer1D(nn.Module):

    """
        OrthogonalLayer1D make the outputs of each unit of the previous sub-layer orthogonal to each other.
        Orthogonalization is performed using Gram-Schmidt orthogonalization.
    """

    def __init__(self):
        super(OrthogonalLayer1D, self).__init__()

    def forward(self,x):

        """
        Arg:
            x: The parallel formated input with shape: [n_models,n_samples,dim]

        return:
            basis: Orthogonalized version of the input (x). The shape of basis is [n_models,n_samples,dim].
                   For each sample, the outputs of all of the models (n_models) will be orthogonal
                   to each other.
        """


        x1 = torch.transpose(x, 0,1)
        basis = torch.unsqueeze(x1[:, 0, :] / (torch.unsqueeze(torch.linalg.norm(x1[:, 0, :], axis=1), 1)), 1)

        for i in range(1, x1.shape[1]):
            v = x1[:, i, :]
            v = torch.unsqueeze(v, 1)
            w = v - torch.matmul(torch.matmul(v, torch.transpose(basis, 2, 1)), basis)
            wnorm = w / (torch.unsqueeze(torch.linalg.norm(w, axis=2), 2))
            basis = torch.cat([basis, wnorm], axis=1)

        basis = torch.transpose(basis,0,1)
        return basis


class OrthogonalLayer2D(nn.Module):

    """
        OrthogonalLayer2D make the filter outputs of each unit of the previous convolutional sub-layer
        orthogonal to each other.
        Orthogonalization is performed using Gram-Schmidt orthogonalization.
        Here the orthogonalization is applied at the filter level of convolution, meaning that for each
        sample, the output of corresponding filters of all of the individual models will be orthogonalized
        to each other.
    """

    def __init__(self):
        super(OrthogonalLayer2D, self).__init__()

    def forward(self,x):

        """
        Arg:
            x: The parallel formated input with shape: [n_samples,n_models,n_channels,im_hight,im_width]

        return:
            basis: Orthogonalized version of the input (x).
                   The shape of basis is [n_samples,n_models,n_channels,im_hight,im_width].
                   For each sample, and each filter, the outputs of all of the models (n_models) will be orthogonal
                   to each other.
        """

        if x.ndim==5:
            im_height = x.shape[3]
            im_width = x.shape[4]
            x = torch.flatten(x,3)
            x = torch.transpose(x,0,1)
            x = torch.transpose(x,1,2)


        norm = torch.unsqueeze( (torch.norm(x[:, :, 0, :], dim=2)) , axis=2)
        basis = torch.unsqueeze(
            x[:, :, 0, :] / norm, dim=2)
        basis[torch.isnan(basis)] = 0

        n_models = x.shape[2]
        for i in range(1, n_models):
            v = x[:, :, i, :]

            v = torch.unsqueeze(v, 2)
            m1 = torch.matmul(v, torch.transpose(basis, 3,2))
            m2 = torch.matmul(m1, basis)
            w = v -  m2

            # wnorm = w
            norm_w = torch.unsqueeze( (torch.norm(w, dim=3)), 3) #+ 1e-100
            wnorm = w / norm_w
            wnorm[torch.isnan(wnorm)] = 0
            basis = torch.cat([basis, wnorm], axis=2)

        basis = torch.transpose(basis, 1, 2)
        basis = torch.transpose(basis, 0, 1)
        basis = torch.reshape(basis,[n_models,basis.shape[1],basis.shape[2],im_height,im_width])
        return basis



class AggregationLayer(nn.Module):

    """
        AggregationLayer aggregate multiple parallel models with each other.
        The result of the aggregation will be later fed to the meta part of the ensemble.
    """

    def __init__(self,stride=2,input_dim=2,output_dim=1):

        """
        Args:
             stride: is used to determine number of based models to be aggregated with each other.
             input_dim: determines the dimension of recieved input.
             output_dim: determines the dimension of output of the aggregation layer.
        """

        super(AggregationLayer, self).__init__()
        self.stride = stride
        self.n_models = n_models_global
        self.n_models_left = 0
        global agg_out_dim
        agg_out_dim = int(n_models_global/stride)*output_dim
        if np.mod(n_models_global,stride)!=0:
            agg_out_dim += output_dim
        self.output_dim = output_dim
        self.aggs = []
        for i in range(0, self.n_models, self.stride):
            if i >= self.n_models - self.stride + 1:
                self.n_models_left = self.n_models - i
                break
            self.aggs.append(nn.Linear(self.stride*input_dim, self.output_dim))

        if self.n_models_left != 0:
          self.aggs.append(nn.Linear(self.n_models_left * input_dim, self.output_dim))
        self.aggs = nn.ModuleList(self.aggs)

    def forward(self,x):

        """
        Arg:
            x: is the intput received from the last layer of the base models with the shape: [n_models,n_samples,dim].

        return:
            aggregated_out: is the output generated by the aggregation layer
        """

        aggregated_in_list = []
        for j in range(self.stride):
            aggregated_in_list.append(x[j])
        aggregated_in = torch.cat(aggregated_in_list, axis=1)
        aggregated_out = F.relu(self.aggs[0](aggregated_in))

        counter = 1
        for i in range(self.stride,self.n_models,self.stride):
            if i >= self.n_models - self.stride + 1:
                n_models_left = self.n_models - i
                break

            aggregated_in_list = []
            for j in range(self.stride):
                try:
                    aggregated_in_list.append(x[i+j])
                except:
                    temp = 1
            aggregated_in = torch.cat(aggregated_in_list, axis=1)

            aggregated_out_cuurent = F.relu(self.aggs[counter](aggregated_in))
            aggregated_out = torch.cat((aggregated_out,aggregated_out_cuurent),axis=1)
            counter += 1

        if self.n_models_left!=0:
            aggregated_in_list = []
            for j in range(self.n_models_left):
                aggregated_in_list.append(x[i + j])
            aggregated_in = torch.cat(aggregated_in_list, axis=1)
            aggregated_out_cuurent = F.relu(self.aggs[counter](aggregated_in))
            aggregated_out = torch.cat((aggregated_out, aggregated_out_cuurent), axis=1)
        return aggregated_out
