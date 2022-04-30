import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Lambda

class MnistDataSet:
    def __init__(self) -> None:
        pass

    def mnist_dataset(self):
        # loading dataset
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalizing images
        train_images = train_images/255.0
        test_images = test_images/255.0
        return train_images, train_labels, test_images, test_labels

class ModelWithLambdaLayerLamdbaFunction:
    def __init__(self) -> None:
        pass

    def model(self):
        model = Sequential(
            [
                Flatten(input_shape=(28, 28)),
                Dense(128),
                Lambda(lambda x: tf.abs(x)),
                Dense(10, activation='softmax')
            ]
        )

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

class ModelWithLambdaLayercustomFunction:
    def __init__(self) -> None:
        pass
    
    def my_relu(self, x):
        return K.maximum(x, 0)

    def model(self):
        model = Sequential(
            [
                Flatten(input_shape=(28, 28)),
                Dense(128),
                Lambda(self.my_relu),
                Dense(10, activation='softmax')
            ]
        )

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = MnistDataSet().mnist_dataset()
    lamdba_func_model = ModelWithLambdaLayerLamdbaFunction().model()
    lamdba_func_model.fit(train_images, train_labels, epochs=5)
    print(lamdba_func_model.evaluate(test_images, test_labels))

    lamdba_func_model = ModelWithLambdaLayercustomFunction().model()
    lamdba_func_model.fit(train_images, train_labels, epochs=5)
    print(lamdba_func_model.evaluate(test_images, test_labels))
