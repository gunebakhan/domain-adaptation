# Cropland classification

Semantic segmentation and domain adaptation in remote sensing using SAR images

## Usage

```python
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

```

## Example
Notebook files are examples of domain adaptive segmentation methods.
DAUNet is domain adaptive unet model with GRL.
Siamese GAN is another method for domain adaptation using UNet for segmentation.