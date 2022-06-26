from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Add, MaxPool2D, Dropout
from keras.models import Model

class UNet:
    def __init__(self, inputs, output_channels) -> None:
        self._inputs = inputs
        self._output_channels = output_channels

    def conv2d_block(self, input_tensor, n_filters, kernel_size=(3, 3)):
        '''
        Adds 2 convolutional layers with the parameters passed to it

        Args:
            input_tensor (tensor) -- the input tensor
            n_filters (int) -- number of filters
            kernel_size (int) -- kernel size for the convolution

        Returns:
            tensor of output features
        '''
        x = input_tensor
        for _ in range(2):
            x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', 
                       kernel_initializer = 'he_normal', activation='relu')(x)
        return x
    
    def encoder_block(self, inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
        '''
        Adds two convolutional blocks and then perform down sampling on output of convolutions.

        Args:
            input_tensor (tensor) -- the input tensor
            n_filters (int) -- number of filters
            kernel_size (int) -- kernel size for the convolution

        Returns:
            f - the output features of the convolution block 
            p - the maxpooled features with dropout
        '''
        f = self.conv2d_block(inputs, n_filters)
        p = MaxPool2D(pool_size)(f)
        p = Dropout(dropout)(p)
        return f, p
        
    def encoder(self, inputs):
        f1, p1 = self.encoder_block(inputs, 64)
        f2, p2 = self.encoder_block(p1, 128)
        f3, p3 = self.encoder_block(p2, 256)
        f4, p4 = self.encoder_block(p3, 512)
        return p4, (f1, f2, f3, f4)
    
    def bottleneck(self, inputs):
        '''
        This function defines the bottleneck convolutions to extract more features before the upsampling layers.
        '''
        
        bottle_neck = self.conv2d_block(inputs, n_filters=1024)
        return bottle_neck

    def decoder_block(self, inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
        '''
        defines the one decoder block of the UNet

        Args:
            inputs (tensor) -- batch of input features
            conv_output (tensor) -- features from an encoder block
            n_filters (int) -- number of filters
            kernel_size (int) -- kernel size
            strides (int) -- strides for the deconvolution/upsampling
            padding (string) -- "same" or "valid", tells if shape will be preserved by zero padding

        Returns:
            c (tensor) -- output features of the decoder block
        '''
        u = Conv2DTranspose(n_filters, kernel_size=kernel_size, strides=strides, padding="same")(inputs)
        c = concatenate([u, conv_output])
        c = Dropout(dropout)(c)
        c = self.conv2d_block(c, n_filters, kernel_size=3)
        return c


        
        
    def decoder(self, inputs, convs, output_channels):
        '''
        Defines the decoder of the UNet chaining together 4 decoder blocks. 
        
        Args:
            inputs (tensor) -- batch of input features
            convs (tuple) -- features from the encoder blocks
            output_channels (int) -- number of classes in the label map

        Returns:
            outputs (tensor) -- the pixel wise label map of the image
        '''
        
        f1, f2, f3, f4 = convs

        c6 = self.decoder_block(inputs, f4, n_filters=512, kernel_size=(3,3), strides=(2,2), dropout=0.3)
        c7 = self.decoder_block(c6, f3, n_filters=256, kernel_size=(3,3), strides=(2,2), dropout=0.3)
        c8 = self.decoder_block(c7, f2, n_filters=128, kernel_size=(3,3), strides=(2,2), dropout=0.3)
        c9 = self.decoder_block(c8, f1, n_filters=64, kernel_size=(3,3), strides=(2,2), dropout=0.3)

        outputs = Conv2D(output_channels, (1, 1), activation='softmax')(c9)

        return outputs
    
    def unet(self):
        encoder_output, convs = self.encoder(self._inputs)
        bottle_neck_output = self.bottleneck(encoder_output)
        outputs = self.decoder(bottle_neck_output, convs, output_channels=self._output_channels)
        model = Model(inputs=self._inputs, outputs=outputs)
        return model



if __name__ == '__main__':
    input = Input(shape=(128, 128, 3))
    output_channels = 3

    # Create UNet model
    unet_model = UNet(input, output_channels).unet() 
    print(unet_model.summary())