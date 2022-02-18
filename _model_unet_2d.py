
from __future__ import absolute_import

from layer_utils import *
from activations import GELU, Snake
from _model_domain_adaptation import domain_adaptation   

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU', 
              pool=True, batch_norm=False, name='left0'):
    '''
    The encoder block of U-net.
    
    UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU', 
              pool=True, batch_norm=False, name='left0')
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    pool_size = 2
    
    X = encode_layer(X, channel, pool_size, pool, activation=activation, 
                     batch_norm=batch_norm, name='{}_encode'.format(name))

    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv'.format(name))
    
    return X


def UNET_right(X, X_list, channel, kernel_size=3, 
               stack_num=2, activation='ReLU',
               unpool=True, batch_norm=False, concat=True, name='right0'):
    
    '''
    The decoder block of U-net.
    
    Input
    ----------
        X: input tensor.
        X_list: a list of other tensors that connected to the input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of convolutional layers.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        concat: True for concatenating the corresponded X_list elements.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    
    pool_size = 2
    
    X = decode_layer(X, channel, pool_size, unpool, 
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))
    
    # linear convolutional layers before concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, 
                   batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))
    if concat:
        # <--- *stacked convolutional can be applied here
        X = concatenate([X,]+X_list, axis=3, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation, 
                   batch_norm=batch_norm, name=name+'_conv_after_concat')
    
    return X

def unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True,  
                 conv_type='Conv2D', siamese=False, name='unet'):
    
    '''
    
    unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                 activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                 conv_type='Conv2D', siamese=False, name='unet')
    
    ----------
    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. 
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        conv_type: Choose between Conv2D and ConvLSTM2D
        siamese: If True use siamese domain adaptation architecture
        name: prefix of the created keras model and its layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    
    activation_func = eval(activation)

    X_skip = []
    depth_ = len(filter_num)


    X = input_tensor

    # stacked conv2d before downsampling
    X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation, 
                    conv_type=conv_type, batch_norm=batch_norm, name='{}_down0'.format(name))
    X_skip.append(X)

    # downsampling blocks
    for i, f in enumerate(filter_num[1:]):
        X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=pool, 
                        batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))        
        X_skip.append(X)

    # reverse indexing encoded feature maps
    X_skip = X_skip[::-1]
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    # bottleneck for using it in domain adaptation part
    bottle_neck = X_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)

    # reverse indexing filter numbers
    filter_num_decode = filter_num[:-1][::-1]

    # upsampling with concatenation
    if siamese:
        X1 = X
        X2 = X
        for i in range(depth_decode):
            X1 = UNET_right(X1, [X_decode[i],], filter_num_decode[i], stack_num=stack_num_up, activation=activation, 
                       unpool=unpool, batch_norm=batch_norm, name='{}_up{}_segmentor'.format(name, i))

        # if tensors for concatenation is not enough
        # then use upsampling without concatenation 
        if depth_decode < depth_-1:
            for i in range(depth_-depth_decode-1):
                i_real = i + depth_decode
                X1 = UNET_right(X1, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation, 
                       unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}_segmentor'.format(name, i_real))   
            
        for i in range(depth_decode):
            X2 = UNET_right(X2, [X_decode[i],], filter_num_decode[i], stack_num=stack_num_up, activation=activation, 
                       unpool=unpool, batch_norm=batch_norm, name='{}_up{}_reconst'.format(name, i))

        # if tensors for concatenation is not enough
        # then use upsampling without concatenation 
        if depth_decode < depth_-1:
            for i in range(depth_-depth_decode-1):
                i_real = i + depth_decode
                X2 = UNET_right(X2, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation, 
                       unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}_reconst'.format(name, i_real))  
        
        return bottle_neck, X1, X2
    else:
        for i in range(depth_decode):
            X = UNET_right(X, [X_decode[i],], filter_num_decode[i], stack_num=stack_num_up, activation=activation, 
                        unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

        # if tensors for concatenation is not enough
        # then use upsampling without concatenation 
        if depth_decode < depth_-1:
            for i in range(depth_-depth_decode-1):
                i_real = i + depth_decode
                X = UNET_right(X, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation, 
                        unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}'.format(name, i_real))   
        return bottle_neck, X

def unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
            conv_type='Conv2D', name='unet', siamese=False,
            is_domain_adaptation=False, da_type='conv', da_kernels=None):
    '''
    
    unet_2d(input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
            activation='ReLU', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
            conv_type='Conv2D', name='unet', siamese=False,
            is_domain_adaptation=False, da_type='conv', da_kernels=None):
    
    ----------
    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. 
    In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        n_labels: number of output labels.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                           Default option is 'Softmax'.
                           if None is received, then linear activation is applied.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.                 
        name: prefix of the created keras model and its layers.
        siamese: If True use siamese domain adaptation architecture
        is_domain_adaptation: Apply domain adaptation if it is True.
        da_type: Type of domain adaptation, conv2d layers or using histograms.
        da_kernels: kernel sizes if domain adaptation kernels type is conv2d
        
        
    Output
    ----------
        model: a keras model for segmentation, 
            [OPTIONAL:a keras model for domain adaptation based on GRL layer],
            [OPTIONAL:a keras model for domain adaptation based on siamese networks with another model wich its output is bottleneck layer]
    
    '''
    activation_func = eval(activation)

        
    IN = Input(input_size)
    
    if siamese:
        bottle_neck, X1, X2 = unet_2d_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                     activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool,
                     conv_type=conv_type, name=name)
                     
    
        # output layer
        OUT = CONV_output(X1, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
        DOMAIN_OUTPUT = domain_adaptation(X1)
        RECONST = CONV_output(X2, 12, kernel_size=1, activation="Sigmoid", name='{}_output'.format('reconstruction'))
    
    
        # functional API unet-model
        segmentor = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
        domain_clf = Model(inputs=[IN,], outputs=[DOMAIN_OUTPUT,], name="domain_classifier")
        reconstruction_model = Model(inputs=[IN,], outputs=[RECONST,], name="reconstruction")
        feature_extractor = Model(inputs=[IN,], outputs=[bottle_neck,], name="feature_extractor")
        # functional API DA-model

    
        return segmentor, domain_clf, reconstruction_model, feature_extractor
        
    # base    
    bottle_neck, X = unet_2d_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                     activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool,
                     conv_type=conv_type, name=name)
    
    # output layer
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    
    if is_domain_adaptation:
        DA_OUT = domain_adaptation(bottle_neck, da_type, da_kernels)
    
    # functional API unet-model
    model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    # functional API DA-model
    if is_domain_adaptation:
        da_model = Model(inputs=[IN,], outputs=[DA_OUT,], name="DA_model")
        return da_model, model
    
    return None, model
