import tensorflow as tf
from tensorflow.keras.layers import Layer

class Sampling(Layer):
    def call(self, inputs):
        """Generates a random sample and combines with the encoder output

        Args:
        inputs -- output tensor from the encoder

        Returns:
        `inputs` tensors combined with a random sample
        """
        # unpack the output of the encoder
        mu, sigma = inputs

        # get the size and dimensions of the batch
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        # Generate a random tensor
        epsilon = tf.random.normal(shape=(batch, dim))

        # Combine the inputs and noise
        return mu + tf.exp(0.5 * sigma) * epsilon
