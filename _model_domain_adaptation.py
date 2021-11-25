import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from layer_utils import CONV_stack


# Gradient Reversal Layer
@tf.custom_gradient
def reverse_gradient(x):
  y = tf.identity(x)
  def grad(dy):
    dy = -4.0 * dy
    return dy
  return y, grad


class GradientReversal(tf.keras.layers.Layer):
  '''Flip the sign of gradient during training.'''
  def __init__(self, **kwargs):
    super(GradientReversal, self).__init__(**kwargs)
    self.supports_masking = False

  def call(self, x, mask=None):
    return reverse_gradient(x)


class BiasInit1(tf.keras.initializers.Initializer):

    def __init__(self, inputs):
        self.inputs = inputs


    def __call__(self, shape, dtype=None, **kwargs):
        print('1:', shape, self.inputs.shape)
        return self.inputs

    def get_config(self):  # To support serialization
        return {"inputs": self.inputs}

    
class BiasInit2(tf.keras.initializers.Initializer):

    def __init__(self, inputs):
        self.inputs = inputs


    def __call__(self, shape, dtype=None, **kwargs):
        print('2:', shape)
        return tf.ones(shape) * self.inputs

    def get_config(self):  # To support serialization
        return {"inputs": self.inputs}


class HistLayer(layers.Layer):
    def __init__(self, in_channels, num_bins=4, two_d=False):
        super(HistLayer, self).__init__()
        
        # histogram data
        self.in_channels = in_channels
        self.num_bins = num_bins
        self.learnable = False
        bin_edges = tf.linspace(-0.05, 1.05, num_bins+1, name=None, axis=0)
        centers = bin_edges + (bin_edges[2] - bin_edges[1]) / 2 # repeat number of centers, diff of groups in keras and pytorch
        self.centers = centers[:-1]
        self.centers = tf.tile(self.centers, [self.in_channels])
        print(self.centers.shape)
        self.width = (bin_edges[2] - bin_edges[1]) / 2
        # self.width = tf.repeat(self.width, self.in_channels*self.num_bins)
        self.two_d = two_d
        
        # Prepare NN layers for histogram computation
        k_initializer_1 = tf.keras.initializers.Constant(1.)
        b_initializer_1 = BiasInit1(-self.centers)
        self.bin_centers_conv = layers.Conv2D(filters=self.num_bins*self.in_channels,
                                             kernel_size=1,
                                             groups=self.in_channels,
                                             kernel_initializer=k_initializer_1,
                                             bias_initializer=b_initializer_1)
        
        k_initializer_2 = tf.keras.initializers.Constant(-1.)
        b_initializer_2 = BiasInit2(self.width)
        self.bin_width_conv = layers.Conv2D(filters=self.num_bins*self.in_channels,
                                           kernel_size=1,
                                           groups=self.num_bins*self.in_channels,
                                           kernel_initializer=k_initializer_2,
                                           bias_initializer=b_initializer_2)
        # self.reshape = layers.Reshape(target_shape=(32, -1))
        self.flatten = layers.Flatten()
#         print(self.bin_centers_conv.get_weights())
#         self.centers = self.bin_centers_conv.get_weights()[1]
#         self.width = self.bin_centers_conv.get_weights()[0]
        self.threshold = tf.keras.layers.ThresholdedReLU(theta=1.0)
        self.hist_pool = tfa.layers.AdaptiveAveragePooling2D(1)

        
    
    def call(self, inputs, normalize=True):
        """Computes differentiable histogram.
        Args:
            inputs: input image
        Returns:
            flattened and un-flattened histogram.
        """
        # |x_i - u_k|
        xx = self.bin_centers_conv(inputs)
        xx = tf.abs(xx)
        
        # w_k - |x_i - u_k|
        xx = self.bin_width_conv(xx)
        
        # 1.01^(w_k -|x_i - u_k|)
        xx = tf.pow(tf.ones_like(xx)*1.01, xx)
        
        # Φ(1.01^(w_k - |x_i - u_k|), 1, 0)
        xx = self.threshold(xx)
        
        # clean-up
        # two_d = self.reshape(xx)
        if normalize:
            xx = self.hist_pool(xx)
        
        else:
            xx = xx.sum([2, 3])
        
        one_d = self.flatten(xx)
        
        return one_d# , two_d

        
        


# Domain Classifier
def domain_adaptation(bottle_neck, da_type='conv2d', da_kernels=None, in_channels=64):

    X = GradientReversal()(bottle_neck)

    if da_type == 'conv2d':
        for i, kernel in enumerate(da_kernels):
            X = CONV_stack(X, channel=kernel, kernel_size=3, stack_num=1, 
               dilation_rate=1, activation='ReLU',
               conv_type='Conv2D',
               batch_norm=True, name=f'da_conv_stack_{i}')
        
        X = layers.GlobalAveragePooling2D()(X)
        outputs = layers.Dense(units=1, activation='sigmoid')(X)
        return outputs

    else:
        X = HistLayer(in_channels, num_bins=128)(X)
        X = tf.keras.layers.Dense(units=32)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dense(units=32, activation='relu')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(X)
        return outputs
