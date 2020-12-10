import tensorflow as tf

from qmc.tf.layers import QMeasureDensityEig
from qmc.tf.tl.layers import BOWL1L2

class QMKDClassifierSGD_BOWL1L2(tf.keras.Model):
    """
    A Quantum Measurement Kernel Density Classifier model trainable using
    gradient descent.

    Arguments:
        input_dim: dimension of the input
        dim_x: dimension of the input quantum feature map
        num_classes: number of classes
        num_eig: Number of eigenvectors used to represent the density matrix. 
                 a value of 0 or less implies num_eig = dim_x
        gamma: float. Gamma parameter of the RBF kernel to be approximated
        random_state: random number generator seed
    """
    def __init__(self, input_dim, dim_x, num_classes, num_eig=0, gamma=1, random_state=None):
        super(QMKDClassifierSGD_BOWL1L2, self).__init__()
        self.fm_x = layers.BOWL1L2Layer(
            input_dim=input_dim)
        self.dim_x = dim_x
        self.num_classes = num_classes
        self.qmd = []
        for _ in range(num_classes):
            self.qmd.append(layers.QMeasureDensityEig(dim_x, num_eig))
        self.gamma = gamma
        self.random_state = random_state

    def call(self, inputs):
        psi_x = self.fm_x(inputs)
        probs = []
        for i in range(self.num_classes):
            probs.append(self.qmd[i](psi_x))
        posteriors = tf.stack(probs, axis=-1)
        posteriors = (posteriors / 
                      tf.expand_dims(tf.reduce_sum(posteriors, axis=-1), axis=-1))
        return posteriors

    def set_rhos(self, rhos):
        for i in range(self.num_classes):
            self.qmd[i].set_rho(rhos[i])
        return

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "num_classes": self.num_classes,
            "num_eig": self.num_eig,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}