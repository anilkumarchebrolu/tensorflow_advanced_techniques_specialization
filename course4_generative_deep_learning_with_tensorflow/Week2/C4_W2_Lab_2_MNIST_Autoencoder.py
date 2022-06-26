import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense

def map_image(image, label):
    '''Normalizes and flattens the image. Returns image as input and label.'''
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    image = tf.reshape(image, shape=(784,))

    return image, image

def mnist_dataset():
    # Load the train and test sets from TFDS
    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 1024

    train_dataset = tfds.load('mnist', as_supervised=True, split="train")
    train_dataset = train_dataset.map(map_image)
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    test_dataset = tfds.load('mnist', as_supervised=True, split="test")
    test_dataset = test_dataset.map(map_image)
    test_dataset = test_dataset.batch(BATCH_SIZE).repeat()
    return train_dataset, test_dataset

def simple_autoencoder(inputs):
    '''Builds the encoder and decoder using Dense layers.'''
    encoder = Dense(32, activation='relu')(inputs)
    decoder = Dense(784, activation='sigmoid')(encoder)
    return encoder, decoder


def deep_autoencoder(inputs):
  '''
    Builds the encoder and decoder using Dense layers.
    --> Stacked Auto Encoder
  '''
  
  encoder = tf.keras.layers.Dense(units=128, activation='relu')(inputs)
  encoder = tf.keras.layers.Dense(units=64, activation='relu')(encoder)
  encoder = tf.keras.layers.Dense(units=32, activation='relu')(encoder)

  decoder = tf.keras.layers.Dense(units=64, activation='relu')(encoder)
  decoder = tf.keras.layers.Dense(units=128, activation='relu')(decoder)
  decoder = tf.keras.layers.Dense(units=784, activation='sigmoid')(decoder)
  
  return encoder, decoder

def display_one_row(disp_images, offset, shape=(28, 28)):
  '''Display sample outputs in one row.'''
  for idx, test_image in enumerate(disp_images):
    plt.subplot(3, 10, offset + idx + 1)
    plt.xticks([])
    plt.yticks([])
    test_image = np.reshape(test_image, shape)
    plt.imshow(test_image, cmap='gray')


def display_results(disp_input_images, disp_encoded, disp_predicted, enc_shape=(8,4)):
  '''Displays the input, encoded, and decoded output values.'''
  plt.figure(figsize=(15, 5))
  display_one_row(disp_input_images, 0, shape=(28,28,))
  display_one_row(disp_encoded, 10, shape=enc_shape)
  display_one_row(disp_predicted, 20, shape=(28,28,))

if __name__ == '__main__':
    BATCH_SIZE = 128
    train_dataset, test_dataset = mnist_dataset()
    # set the input shape
    inputs = tf.keras.layers.Input(shape=(784,))

    # get the encoder and decoder output
    encoder_output, decoder_output = simple_autoencoder(inputs)

    # setup the encoder because you will visualize its output later
    encoder_model = tf.keras.Model(inputs=inputs, outputs=encoder_output)

    # setup the autoencoder
    autoencoder_model = tf.keras.Model(inputs=inputs, outputs=decoder_output)

    # compile the model
    autoencoder_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')

    # train the model
    train_steps = 60000 // BATCH_SIZE
    simple_auto_history = autoencoder_model.fit(train_dataset, steps_per_epoch=train_steps, epochs=50)

    # take 1 batch of the dataset
    test_dataset = test_dataset.take(1)

    # take the input images and put them in a list
    output_samples = []
    for input_image, image in tfds.as_numpy(test_dataset):
        output_samples = input_image

    # pick 10 random numbers to be used as indices to the list above
    idxs = np.random.choice(BATCH_SIZE, size=10)

    # get the encoder output
    encoded_predicted = encoder_model.predict(test_dataset)

    # get a prediction for the test batch
    simple_predicted = autoencoder_model.predict(test_dataset)

    # display the 10 samples, encodings and decoded values!
    display_results(output_samples[idxs], encoded_predicted[idxs], simple_predicted[idxs])
    print("end")