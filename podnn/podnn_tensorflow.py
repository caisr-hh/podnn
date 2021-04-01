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

This file provides five different classes for creating PODNN architectures in Tensorflow.

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
import tensorflow as tf
from tensorflow.keras import layers, Model
from copy import deepcopy,copy


n_models_global = 4
agg_input_dim = 100
agg_out_dim = 3
im_height = 32
im_width = 32

class InputLayer(layers.Layer):

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

    def call(self,x):

        """
            Arg
              x: is the input to the network as in other standard deep neural networks.

            return:
              x_parallel: is the parallel form of the received input (x).
        """

        x_parallel = tf.expand_dims(x,0)
        x_parallel_next = tf.expand_dims(x, 0)
        for i in range(1,self.n_models):
            x_parallel =  tf.concat((x_parallel,x_parallel_next),axis=0)

        return x_parallel


class UnitModel(layers.Layer):

    """
        UnitModel perform computations defined by the modules it received on the input.
    """

    def __init__(self,unit_model):

        """
        Arg:
            unit_model: is a list of moduls specifies what computational meach unit of the parallel layer contains.
                        unit_model is a number of layer definitions followed by each other.
        """

        super(UnitModel, self).__init__()
        self.layers = []
        for i in range(len(unit_model)):
            self.layers.append(copy(unit_model[i]))


    def call(self, x):

        """
        Arg:
            x: is the parallel input with shape [n_samples,dim] for fully connected layers and
                                                [n_samples,im_hight,im_width,n_channels] for convlutional layers.

        return:
            unit_output: is the output formed by applying modules within each units on the input.

            shape: [n_samples,out_dim] for fully connected layers and
                   [n_samples,out_im_hight,out_im_width,n_channels] for convlutional layers.

            note that out_dim, out_im_hight, and out_im_width are determined by the computation
            defined with modules within unit_model.

        """

        unit_output = x
        for i in range(len(self.layers)):
            unit_output = self.layers[i](unit_output)

        return unit_output


class ParallelLayer(layers.Layer):

    """
        Parallellayer creates a parallel layer from the modules of unit_model it receives.
    """

    def __init__(self,unit_model):

        """
        Arg:
            unit_model: specifies what computational module each unit of the parallel layer contains.
                        unit_model is a number of layer definitions followed by each other.
        """

        super(ParallelLayer, self).__init__()

        self.n_models = n_models_global
        self.model_layers = []

        for i in range(self.n_models):

            self.model_layers.append(UnitModel(unit_model))


    def call(self, x):

        """
        Arg:
            x: is the parallel input with shape [n_models,n_samples,dim] for fully connected layers and
                                                [n_models,n_samples,im_hight,im_width,n_channels] for convlutional layers.

        return:
            parallel_output: is the output formed by applying modules within each units on the input.

            shape: [n_models,n_samples,dim] for fully connected layers and
                   [n_models,n_samples,im_hight,im_width,n_channels] for convlutional layers.

            note that out_dim, out_im_hight, and out_im_width are determined by the computation
            defined with modules within unit_model.

        """


        parallel_output = (self.model_layers[0](x[0]))
        parallel_output = tf.expand_dims(parallel_output, 0)
        for i in range(1, self.n_models):

            next_layer = self.model_layers[i](x[i])
            next_layer = tf.expand_dims(next_layer, 0)
            parallel_output = tf.concat((parallel_output, next_layer), 0)

        return parallel_output


class OrthogonalLayer1D(layers.Layer):

    """
        OrthogonalLayer1D make the outputs of each unit of the previous sub-layer orthogonal to each other.
        Orthogonalization is performed using Gram-Schmidt orthogonalization.
    """

    def __init__(self):
        super(OrthogonalLayer1D,self).__init__()

    def call(self,inputs):

            """
            Arg:
               x: The parallel formated input with shape: [n_models,n_samples,dim]

            return:
               basis: Orthogonalized version of the input (x). The shape of basis is [n_models,n_samples,dim].
                      For each sample, the outputs of all of the models (n_models) will be orthogonal
                      to each other.
            """


            inputs = tf.transpose(inputs,[1,0,2])
            basis = tf.expand_dims(tf.math.divide_no_nan(inputs[:, 0, :],\
                                    tf.expand_dims(tf.norm(inputs[:, 0, :], axis=1), axis=1)), axis=1)
            basis = tf.identity(basis, name="basis")
            for i in range(1, inputs.get_shape()[1]):
                v = inputs[:, i, :]

                v = tf.expand_dims(v, 1)
                w = v - tf.matmul(tf.matmul(v, tf.transpose(basis, [0, 2, 1])), basis)

                #wnorm = w
                wnorm = tf.math.divide_no_nan(w,tf.expand_dims(tf.norm(w,axis=2),2))
                basis = tf.concat([basis, wnorm], axis=1)

            basis = tf.transpose(basis,[1,0,2])

            return basis


class OrthogonalLayer2D(layers.Layer):

    """
        OrthogonalLayer2D make the filter outputs of each unit of the previous convolutional sub-layer
        orthogonal to each other.
        Orthogonalization is performed using Gram-Schmidt orthogonalization.
        Here the orthogonalization is applied at the filter level of convolution, meaning that for each
        sample, the output of corresponding filters of all of the individual models will be orthogonalized
        to each other.
    """

    def __init__(self, layer_type='dense', n_models=2, n_neurons=10, input_dim=3):
        super(OrthogonalLayer2D, self).__init__()

    def call(self,x):

        """
        Arg:
            x: The parallel formated input with shape: [n_samples,n_models,im_hight,im_width,n_channels]

        return:
            basis: Orthogonalized version of the input (x).
                   The shape of basis is [n_samples,n_models,im_hight,im_width,n_channels].
                   For each sample, and each filter, the outputs of all of the models (n_models) will be orthogonal
                   to each other.
        """

        if len(x.shape)==5:
            im_height = x.shape[3]
            im_width = x.shape[4]
            x = tf.reshape(x,[x.shape[0],x.shape[1],x.shape[2],x.shape[3]*x.shape[4]])
            x = tf.transpose(x,[1,2,0,3])


        norm = tf.expand_dims( (tf.norm(x[:, :, 0, :], axis=2)) , axis=2)
        basis = tf.expand_dims(tf.math.divide_no_nan(x[:, :, 0, :] , norm), axis=2)

        n_models = x.shape[2]
        for i in range(1, n_models):
            v = x[:, :, i, :]

            v = tf.expand_dims(v, 2)
            m1 = tf.matmul(v, tf.transpose(basis, [0,1,3,2]))
            m2 = tf.matmul(m1, basis)
            w = v -  m2

            norm_w = tf.expand_dims( (tf.norm(w, axis=3)), 3)

            wnorm = tf.math.divide_no_nan(w, norm_w)
            basis = tf.concat([basis, wnorm], axis=2)

        basis = tf.transpose(basis,[2,0,1,3])
        basis = tf.reshape(basis,[n_models,basis.shape[1],basis.shape[2],im_height,im_width])
        return basis



class AggregationLayer(layers.Layer):

    """
        AggregationLayer aggregate multiple parallel models with each other.
        The result of the aggregation will be later fed to the meta part of the ensemble.
    """

    def __init__(self,stride=2,output_dim=1):

        """
        Args:
             stride: is used to determine number of based models to be aggregated with each other.
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
            self.aggs.append(layers.Dense(self.output_dim))

        if self.n_models_left != 0:
          self.aggs.append(layers.Dense(self.output_dim))



    def call(self,x):

        """
        Arg:
            x: is the intput received from the last layer of the base models with the shape: [n_models,n_samples,dim].

        return:
            aggregated_out: is the output generated by the aggregation layer
        """

        aggregated_in_list = []
        for j in range(self.stride):
            aggregated_in_list.append(x[j])
        aggregated_in = tf.concat(aggregated_in_list, axis=1)

        aggregated_out = tf.keras.activations.relu(self.aggs[0](aggregated_in))

        counter = 1
        for i in range(self.stride,self.n_models,self.stride):
            if i >= self.n_models - self.stride + 1:
                self.n_models_left = self.n_models - i
                break

            aggregated_in_list = []
            for j in range(self.stride):
                try:
                    aggregated_in_list.append(x[i+j])
                except:
                    temp = 1
            aggregated_in = tf.concat(aggregated_in_list, axis=1)

            aggregated_out_cuurent = tf.keras.activations.relu(self.aggs[counter](aggregated_in))
            aggregated_out = tf.concat((aggregated_out,aggregated_out_cuurent),axis=1)
            counter += 1

        if self.n_models_left!=0:
            aggregated_in_list = []
            for j in range(self.n_models_left):
                aggregated_in_list.append(x[i + j])
            aggregated_in = tf.concat(aggregated_in_list, axis=1)
            aggregated_out_cuurent = tf.keras.activations.relu(self.aggs[counter](aggregated_in))
            aggregated_out = tf.concat((aggregated_out, aggregated_out_cuurent), axis=1)
        return aggregated_out


