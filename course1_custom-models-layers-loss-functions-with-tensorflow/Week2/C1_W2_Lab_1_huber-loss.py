import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class SimpleDummyDataSet:
    def __init__(self) -> None:
        pass
    
    def simply_dummy_dataset(self):
        '''
        Here Our dummy dataset is just a pair of arrays xs and ys defined by the relationship  𝑦=2𝑥−1 . 
        xs are the inputs while ys are the labels.
        '''
        # inputs
        X = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

        # labels
        Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
        return X, Y


def huber_loss(y_true, y_pred):
    '''
    https://en.wikipedia.org/wiki/Huber_loss
    '''
    delta = 1
    a = y_true - y_pred
    small_error = 0.5 * tf.square(a)
    is_small_error = tf.abs(a) <= delta
    large_error = delta * (tf.abs(a) - (0.5 * delta))
    return tf.where(is_small_error, small_error, large_error)


if __name__ == '__main__':

    # Fetching data
    X, Y = SimpleDummyDataSet().simply_dummy_dataset()

    # Model based on sgd loss
    model = Sequential([Dense(units=1, use_bias=True, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mse')
    model.fit(X, Y, epochs=500, verbose=0)
    print(f"model prediction based on sgd loss {model.predict([10.0])}")

    # Model based on Huber loss
    model = Sequential([Dense(units=1, use_bias=True, input_shape=[1])])
    model.compile(optimizer='sgd', loss=huber_loss)
    model.fit(X, Y, epochs=500, verbose=0)
    print(f"model prediction based on huber loss {model.predict([10.0])}")
