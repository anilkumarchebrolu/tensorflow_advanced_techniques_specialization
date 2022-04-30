import tensorflow as tf
from tensorflow import keras
import numpy as np

class CustomLayer(keras.layers.Layer):
    def __init__(self) -> None:
        super(CustomLayer, self).__init__()
        
        self.my_var = tf.Variable(100)
        self.list_of_vars = [
            tf.Variable(x) for x in range(2)
        ]

    def call(self, inputs):
        return None

if __name__ == '__main__':
    # Evaluating Tensors
    x = 2
    result = tf.square(x)
    print(f"Squared value of {x} is {result}")

    # Broadcasting values
    a = tf.constant([[1, 2], [3, 4]])
    result = tf.add(a, 1) # 1 which is a scalar is broadcasted and added to all the values
    print(f"Addition broadcasted {result}") 

    # operator overloading
    result = a ** 2
    print(f"multiplication operation overloaded {result}")

    # Numpy compatibility for tensorflow.
    a = tf.constant(5)
    b = tf.constant(3)
    result = np.multiply(a, b)
    print(f"numpy multiplication {result}")

    # Numpy interportability
    nd_array = np.ones(shape=(3, 3))
    result = tf.multiply(nd_array, 3)
    numpy_result = result.numpy()
    print(f"Numpy result {numpy_result}")

    # Evaluating variables
    v = tf.Variable(10)
    result = v+1
    print(f"Addition {result}")

    v.assign_add(1)
    print(v.read_value().numpy())
    print(v.numpy())

    # Examining custom layers.
    print("Examining the layers")
    custom_layer = CustomLayer()
    for variable in custom_layer.variables:
        print(variable.numpy())