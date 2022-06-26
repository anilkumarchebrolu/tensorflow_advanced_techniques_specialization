import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from IPython import display
from variational_auto_encoder_model import VariationalAutoEncoder

# Define global constants to be used in this notebook
BATCH_SIZE=128
LATENT_DIM=2

def map_image(image, label):
    '''returns a normalized and reshaped tensor from a given image'''
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    image = tf.reshape(image, shape=(28, 28, 1,))
    
    return image


def get_dataset(map_fn, is_validation=False):
    '''Loads and prepares the mnist dataset from TFDS.'''
    if is_validation:
        split_name = "test"
    else:
        split_name = "train"

    dataset = tfds.load('mnist', as_supervised=True, split=split_name)
    dataset = dataset.map(map_fn)
    
    if is_validation:
        dataset = dataset.batch(BATCH_SIZE)
    else:
        dataset = dataset.shuffle(1024).batch(BATCH_SIZE)

    return dataset


def get_models(input_shape, latent_dim):
    """Returns the encoder, decoder, and vae models"""
    vae = VariationalAutoEncoder()
    encoder, conv_shape = vae.encoder_model(latent_dim=latent_dim, input_shape=input_shape)
    decoder = vae.decoder_model(latent_dim=latent_dim, conv_shape=conv_shape)
    vae = vae.model(encoder, decoder, input_shape=input_shape)
    return encoder, decoder, vae

def generate_and_save_images(model, epoch, step, test_input):
    """Helper function to plot our 16 images

    Args:

    model -- the decoder model
    epoch -- current epoch number during training
    step -- current step number during training
    test_input -- random tensor with shape (16, LATENT_DIM)
    """

    # generate images from the test input
    predictions = model.predict(test_input)

    # plot the results
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    fig.suptitle("epoch: {}, step: {}".format(epoch, step))
    plt.savefig('image_at_epoch_{:04d}_step{:04d}.png'.format(epoch, step))
    plt.show()


if __name__ == '__main__':
    train_dataset = get_dataset(map_image)
    encoder, decoder, vae = get_models(input_shape=(28, 28, 1), latent_dim=LATENT_DIM)

    # Define our loss functions and optimizers
    optimizer = tf.keras.optimizers.Adam()
    loss_metric = tf.keras.metrics.Mean()
    bce_loss = tf.keras.losses.BinaryCrossentropy()

    ## Training loop. 
    # generate random vector as test input to the decoder
    random_vector_for_generation = tf.random.normal(shape=[16, LATENT_DIM])

    # number of epochs
    epochs = 100

    # initialize the helper function to display outputs from an untrained model
    generate_and_save_images(decoder, 0, 0, random_vector_for_generation)

    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))

        # iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:

                # feed a batch to the VAE model
                reconstructed = vae(x_batch_train)

                # compute reconstruction loss
                flattened_inputs = tf.reshape(x_batch_train, shape=[-1])
                flattened_outputs = tf.reshape(reconstructed, shape=[-1])
                loss = bce_loss(flattened_inputs, flattened_outputs) * 784
                
                # add KLD regularization loss
                loss += sum(vae.losses)  

                # get the gradients and update the weights
                grads = tape.gradient(loss, vae.trainable_weights)
                optimizer.apply_gradients(zip(grads, vae.trainable_weights))

                # compute the loss metric
                loss_metric(loss)

                # display outputs every 100 steps
                if step % 100 == 0:
                    display.clear_output(wait=False)    
                    generate_and_save_images(decoder, epoch, step, random_vector_for_generation)
                    print('Epoch: %s step: %s mean loss = %s' % (epoch, step, loss_metric.result().numpy()))