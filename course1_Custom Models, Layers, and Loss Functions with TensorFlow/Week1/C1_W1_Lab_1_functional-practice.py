'''
Objective: To learn to build Functional API.

--> We will build a sequential model.
--> Will build similar model using functional api.
--> Will train the 
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow import nn
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pydot
from tensorflow.python.ops.gen_math_ops import mod


def build_sequential_api():
    '''
    Creating an instance sequential model api

    Returns:
    --------
    Instance of the Sequential model.
    '''

    seq_model = Sequential(
        [
            Flatten(input_shape=(28, 28)),
            Dense(128, activation=nn.relu),
            Dense(10, activation=nn.softmax)
        ]
    )

    return seq_model

def build_functional_api():
    '''
    Creating an instance of keras model using functional api

    Returns:
    --------
    Instance of the keras model created through functional api.
    '''

    # Defining the inputs
    input_layer = Input(shape=(28, 28))

    # Defining the layers
    x = Flatten()(input_layer)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Creating the keras model instance
    func_model = Model(inputs= input_layer, outputs= output_layer)
    return func_model


def model_training_and_evaluation(model):
    # loading dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalizing images
    train_images = train_images/255.0
    test_images = test_images/255.0

    # compiling the model
    model.compile(
        optimizer = Adam(),
        loss = SparseCategoricalCrossentropy(),
        metrics = ['accuracy']
    )

    model.fit(
        train_images, train_labels, epochs=5
    )

    model.evaluate(test_images, test_labels)

if __name__ == '__main__':
    seq_model = build_sequential_api()
    seq_model.summary()

    func_model = build_functional_api()
    func_model.summary()

    model_training_and_evaluation(seq_model)
    model_training_and_evaluation(func_model)
    # issue with the imports 
    plot_model(seq_model, show_shapes=True, show_layer_names=True, to_file='outputs/seq_model.png')
    plot_model(seq_model, show_shapes=True, show_layer_names=True, to_file='outputs/func_model.png')