from sampling import Sampling
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf

class VariationalAutoEncoder:
    
    def encoder_model(self, latent_dim, input_shape):
        """Defines the encoder model with the Sampling layer
        Args:
            latent_dim -- dimensionality of the latent space
            input_shape -- shape of the dataset batch

        Returns:
            model -- the encoder model
            conv_shape -- shape of the features before flattening
        """
        inputs = Input(shape=(input_shape))
        x = Conv2D(32, kernel_size=(3, 3), padding='same', strides=2, activation='relu', name="encode_conv1")(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=(3, 3), padding='same', strides=2, activation='relu', name="encode_conv2")(x)

        # assign to a different variable so you can extract the shape later
        batch_2 = BatchNormalization()(x)

        # flatten the features and feed into the Dense network
        x = Flatten(name="encode_flatten")(batch_2)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)

        mu = Dense(latent_dim, name='latent_mu')(x)
        sigma = Dense(latent_dim, name='latent_sigma')(x)
        
        # feed mu and sigma to the Sampling layer
        z = Sampling()((mu, sigma))

        # building the encoder model
        model = Model(inputs=inputs, outputs=[mu, sigma, z])
        return model, batch_2.shape


    def decoder_model(self, latent_dim, conv_shape):
        """Defines the decoder model.
        Args:
            latent_dim -- dimensionality of the latent space
            conv_shape -- shape of the features before flattening

        Returns:
            model -- the decoder model
        """
        # set the inputs to the shape of the latent space
        inputs = Input(shape=(latent_dim,))
        units = conv_shape[1] * conv_shape[2] * conv_shape[3]

        # feed to a Dense network with units computed from the conv_shape dimensions
        x = Dense(units, activation = 'relu', name="decode_dense1")(inputs)
        x = BatchNormalization()(x)
        
        # reshape output using the conv_shape dimensions
        x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]), name="decode_reshape")(x)

        # upsample the features back to the original dimensions
        x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name="decode_conv2d_2")(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', name="decode_conv2d_3")(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid', name="decode_final")(x)
        model = Model(inputs, x)
        return model

    def kl_reconstruction_loss(self, inputs, outputs, mu, sigma):
        """ Computes the Kullback-Leibler Divergence (KLD)
        Args:
            inputs -- batch from the dataset
            outputs -- output of the Sampling layer
            mu -- mean
            sigma -- standard deviation

        Returns:
            KLD loss
        """
        kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        return kl_loss


    def model(self, encoder, decoder, input_shape):
        """Defines the VAE model
        Args:
            encoder -- the encoder model
            decoder -- the decoder model
            input_shape -- shape of the dataset batch

        Returns:
            the complete VAE model
        """
        # set the inputs
        inputs = encoder.input

        # get mu, sigma, and z from the encoder output
        mu, sigma, z = encoder.outputs

        # get reconstructed output from the decoder
        reconstructed = decoder(z)

        # define the inputs and outputs of the VAE
        model = Model(inputs=inputs, outputs=reconstructed)

        # add the KL loss
        loss = self.kl_reconstruction_loss(inputs, z, mu, sigma)
        model.add_loss(loss)
        return model