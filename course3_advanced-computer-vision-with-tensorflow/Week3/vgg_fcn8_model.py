from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Cropping2D, Add, Activation
from keras.models import Model

class VGGFCN8Model:
    '''
    This function defines the VGG encoder.

    Args:
        image_input (tensor) - batch of images

    Returns:
        tuple of tensors - output of all encoder blocks plus the final convolution layer
    '''
    def __init__(self, input_tensor, num_classes) -> None:
        self._num_filters = 4096
        self._input_tensor = input_tensor
        self._num_classes = num_classes

    def vgg_model(self):
        vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=self._input_tensor)
        f1 = (vgg16.get_layer("block1_pool")).output
        f2 = (vgg16.get_layer("block2_pool")).output
        f3 = (vgg16.get_layer("block3_pool")).output
        f4 = (vgg16.get_layer("block4_pool")).output
        f5 = (vgg16.get_layer("block5_pool")).output

        c6_1 = Conv2D(self._num_filters, kernel_size=(7, 7), activation='relu', padding='same', name='block6_conv1')(f5)
        c6_2 = Conv2D(self._num_filters, kernel_size=(1, 1), activation='relu', padding='same', name='block6_conv2')(c6_1)
        return (f1, f2, f3, f4, c6_2)

    def fcn8_decoder(self, convs):
        '''
        Defines the FCN 8 decoder.

        Args:
            convs (tuple of tensors) - output of the encoder network
            n_classes (int) - number of classes

        Returns:
            tensor with shape (height, width, n_classes) containing class probabilities
        '''
        f1, f2, f3, f4, f5 = convs
        
        # upsample the output of the encoder then crop extra pixels that were introduced
        o = Conv2DTranspose(self._num_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(f5)
        o = Cropping2D(cropping=(1, 1))(o)

        # load the pool 4 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
        o2 = Conv2D(self._num_classes, kernel_size=(1, 1), activation='relu', padding='same')(f4)
        o = Add()([o, o2])

        # upsample the resulting tensor of the operation you just did
        o = Conv2DTranspose(self._num_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
        o = Cropping2D(cropping=(1, 1))(o)

        # load the pool 3 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
        o2 = Conv2D(self._num_classes, kernel_size=(1, 1), activation='relu', padding='same')(f3)
        o = Add()([o, o2])
        
        # upsample up to the size of the original image
        o = Conv2DTranspose(self._num_classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(o)

        # append a softmax to get the class probabilities
        o = (Activation('softmax'))(o)
        return o

    def final_model(self):
        convs = self.vgg_model()
        outputs = self.fcn8_decoder(convs)
        return Model(inputs=self._input_tensor, outputs=outputs)


if __name__ == '__main__':
    inputs = Input(shape=(224,224,3,))
    num_classes = 12
    fcn_model = VGGFCN8Model(inputs, num_classes).final_model()
    fcn_model.summary()