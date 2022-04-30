import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Dense
from tensorflow.python.framework import tensor_shape
import sys
import os
sys.path.append(os.getcwd())
from data.simple_dummy_dataset import SimpleDummyDataSet


class CustomDenseLayer(Layer):
    def __init__(self, units= 32, trainable=True):
        super(CustomDenseLayer, self).__init__(trainable)
        self.units = units

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            name='kernel',
            initial_value = w_init(
                shape=(input_shape[-1], self.units), dtype='float32'
            ),
            trainable=True, 
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            name='bias',
            initial_value=b_init(
                shape=(self.units,), dtype='float32'
            ),
            trainable=True
        )

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        return tf.matmul(inputs, self.w) + self.b

if __name__ == '__main__':
    
    ## understanding custom dense layer
    # Declaring custom dense layer
    """     my_dense = CustomDenseLayer(units=1)

    # define an input and feed into the layer
    x = tf.ones((1, 1))
    y = my_dense(x)

    # printing the variables
    print(y)
    print(my_dense.variables) """

    ## Simple Training
    # loading data
    xs, ys = SimpleDummyDataSet().simply_dummy_dataset()

    # use the Sequential API to build a model with our custom layer
    my_layer = CustomDenseLayer(units=1)
    dense = Dense(units=1)
    input_layer = tf.keras.Input(shape=(1,))
    model = tf.keras.Sequential([input_layer, CustomDenseLayer(1)])

    # configure and train the model
    model.compile(optimizer='sgd', loss='mean_squared_error')
    xs = xs.reshape(-1, 1)
    model.fit(xs, ys, epochs=500,verbose=0)

    # perform inference
    print(model.predict([10.0]))

    # see the updated state of the variables
    print(my_layer.variables)