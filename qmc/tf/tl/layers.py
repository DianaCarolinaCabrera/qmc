import tensorflow as tf
import numpy as np
from typeguard import typechecked
from tensorflow.python.keras import backend as K
from sklearn import preprocessing

from qmc.tf.layers import QFeatureMapRFF

Class BOWL1L2(tf.keras.layers.Layer):
    """BOWL1L2 layer for replace QFeatureMapRFF function.
    Represents the x_vectors after normalization L1, L2 and rebalanced:
    
    Input shape:
        (batch_size, dim_in)
        where dim_x is the dimension of the input state
    Output shape:
        (batch_size, dim)
    Arguments:
        input_dim: int. the dimension of the input
    """
    @typechecked
    def __init__(self, input_dim: int = 1000, **kwargs):
        super(BOWL1L2Layer, self).__init__(**kwargs)
        if input_dim = 0:
            raise NotImplementedError("Dimension of input "
                                      "{}.".format(input_dim))
        self.input_dim = input_dim
        self.ctype = tf.Variable(initial_value = input_dim, 
                                dtype = tf.float32,
                                trainable = False,
                                name = "Change_Type")

    def call(self, inputs):
        return self.ctype(super().call(inputs))
    
    def get_config(self):
        config = {
            "input_dim" = self.input_dim
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    def from_config(cls, config):
        return cls(**config)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    

