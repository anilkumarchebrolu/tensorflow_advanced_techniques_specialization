from datetime import datetime
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import imageio

# Visualization utilities
plt.rc('font', size=20)
plt.rc('figure', figsize=(15, 3))
GIF_PATH = "Week5/outputs/gif.gif"


class SimpleModel:
    def simple_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(16, activation='linear', input_dim=784))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


class DateAndTimeCallback(keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        print(f"Training: batch {batch} begins at {datetime.now().time()}")

    def on_train_batch_end(self, batch, logs=None):
        print(f"Tringing: batch {batch} ends at {datetime.now().time()}")


class DetectOverFittingCallback(keras.callbacks.Callback):
    def __init__(self, threshold=0.7):
        super(DetectOverFittingCallback, self).__init__()
        self.threshold = threshold
    
    def on_epoch_end(self, epoch, logs=None):
        ratio = logs['val_loss']/logs['loss']
        print(f"Epoch: {epoch}, Val/Train loss ratio {ratio}")

        if ratio > self.threshold:
            self.model.stop_training=True


class VisCallback(keras.callbacks.Callback):
    def __init__(self, inputs, ground_truth, disp_frequency=10, n_samples=10):
        super(VisCallback, self).__init__()
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images = []
        self.disp_frequency = disp_frequency
        self.n_samples = n_samples
    
    def display_digits(self, inputs, outputs, ground_truth, epoch, n=10):
        plt.clf()

        plt.yticks([])
        plt.grid(None)
        inputs = np.reshape(inputs, [n, 28, 28])
        inputs = np.swapaxes(inputs, 0, 1)
        inputs = np.reshape(inputs, [28, 28*n])
        plt.imshow(inputs)
        plt.xticks([28*x+14 for x in range(n)], outputs)
        for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
            if outputs[i] == ground_truth[i]: 
                t.set_color('green') 
            else: 
                t.set_color('red')
        plt.grid(None)

    def on_epoch_end(self, epoch, logs=None):
        # Randomly pick n samples
        indices = np.random.choice(len(self.inputs), size=self.n_samples)
        x_test, y_test = self.inputs[indices], self.ground_truth[indices]
        predictions = np.argmax(self.model.predict(x_test), axis=1)

        # display digits 
        self.display_digits(x_test, predictions, y_test, epoch, n= self.n_samples)

        # save the figure
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        self.images.append(np.array(image))

        # Display the digits 
        if epoch % self.disp_frequency == 0:
            plt.show()

    def on_train_end(self, logs=None):
        imageio.mimsave(GIF_PATH, self.images, fps=1)



if __name__ == '__main__':
    # Load example MNIST data and pre-process it
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    
    # Date and Time Callback provides information at each batch.
    simple_model = SimpleModel.simple_model()
    simple_model.fit(x_train, y_train, validation_data=(x_test, y_test),  batch_size=64, epochs=1, steps_per_epoch=5, callbacks=[DateAndTimeCallback()])

    # Lambda Callback example
    lambda_callback = keras.callbacks.LambdaCallback(
        on_epoch_end= lambda epoch, logs: print(f"Epoch: {epoch}, Val/Train loss ratio { logs['val_loss'] / logs['loss'] }")
    )
    simple_model = SimpleModel.simple_model()
    simple_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=5, steps_per_epoch=5, callbacks=[lambda_callback])

    # OverFitting Callback example
    simple_model = SimpleModel.simple_model()
    simple_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=5, callbacks=[DetectOverFittingCallback()])

    # Visualziation Callback example
    simple_model = SimpleModel.simple_model()
    simple_model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=0,  steps_per_epoch=50, callbacks=[VisCallback(x_test, y_test)])