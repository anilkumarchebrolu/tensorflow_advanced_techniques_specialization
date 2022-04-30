import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Layer
import tf2onnx
import onnx

class CNNResidual(Layer):
    '''
    Defining CNNResidual
    '''
    def __init__(self, layers, filters, **kwargs):
        super(CNNResidual, self).__init__()
        self.hidden = [
            Conv2D(filters, (3, 3), activation='relu') for _ in range(layers)
        ]


    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x

class DenseResidual(Layer):
    '''
    Defining CNNResidual
    '''
    def __init__(self, layers, units, **kwargs):
        super(DenseResidual, self).__init__()
        self.hidden = [
            Dense(units, activation='relu') for _ in range(layers)
        ]


    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x

class MyResidual(tf.keras.Model):
    def __init__(self) -> None:
        super(MyResidual, self).__init__()
        self.hidden1 = Dense(30, activation='relu')
        self.conv_block = CNNResidual(2, 32)
        self.dnn_blocks = [
            DenseResidual(2, 32) for _ in range(1, 4)
        ]
        self.out = Dense(1, activation='sigmoid')

    def call(self, input_tensor):
        x = self.hidden1(input_tensor)
        x = self.conv_block(x)
        for dnn_block in self.dnn_blocks:
            x = dnn_block(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    model = MyResidual()