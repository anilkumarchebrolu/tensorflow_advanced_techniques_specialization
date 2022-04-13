# Imports
from os import name
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random


class FashionMnistPairsDataset:
    def __init__(self):
        pass
    
    def create_pairs(self, x, digit_indices):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(10)]) - 1
        
        for d in range(10):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, 10)
                dn = (d + inc) % 10
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
                
        return np.array(pairs), np.array(labels)

    def create_pairs_on_set(self, images, labels):
        
        digit_indices = [np.where(labels == i)[0] for i in range(10)]
        pairs, y = self.create_pairs(images, digit_indices)
        y = y.astype('float32')
        
        return pairs, y


    def show_image(self, image):
        plt.figure()
        plt.imshow(image)
        plt.colorbar()
        plt.grid(False)
        plt.show()

    def fashion_mnist_pairs_dataset(self):
        # load the dataset
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        # prepare train and test sets
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        # normalize values
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        # create pairs on train and test sets
        tr_pairs, tr_y = self.create_pairs_on_set(train_images, train_labels)
        ts_pairs, ts_y = self.create_pairs_on_set(test_images, test_labels)

        return tr_pairs, tr_y, ts_pairs, ts_y


class SiameseNetwork:
    def __init__(self) -> None:
        pass

    def initialize_base_network(self):
        input = Input(shape=(28,28,), name="base_input")
        x = Flatten(name="flatten_input")(input)
        x = Dense(128, activation='relu', name ='first_base_dense')(x)
        x = Dropout(0.2, name='first_dropout')(x)
        x = Dense(128, activation='relu', name ='second_base_dense')(x)
        x = Dropout(0.2, name='second_dropout')(x)
        x = Dense(128, activation='relu', name ='third_base-dense')(x)

        return Model(inputs=input, outputs=x)
    
    def euclidean_distance(self, vectors):
        vec1, vec2 = vectors
        sum_square = K.sum(K.square(vec1 - vec2), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    def eucl_dist_output_shape(self, shapes):
        shape1, _ = shapes
        return (shape1[0], 1)
    
    def plot_model(self, model):
        # plot model graph
        plot_model(model, show_shapes=True, show_layer_names=True, to_file=f'outputs/siamese_network_outputs/outer-model.png')

    def build_siamese_network(self):
        # Intialize base network
        base_network = self.initialize_base_network()
        
        # Create Left and Right networks
        input_left = Input(shape=(28, 28), name="left_input")
        vec_out_left = base_network(input_left)

        input_right = Input(shape=(28, 28), name="right_input")
        vec_out_right = base_network(input_right)

        # Measure the similarity.
        output = Lambda(self.euclidean_distance, name="output_layer", output_shape=self.eucl_dist_output_shape)([vec_out_left, vec_out_right])

        # siamese network
        siamese_model = Model(inputs=[input_left, input_right], outputs=output)
        self.plot_model(siamese_model)
        return siamese_model


class MetricsAndVisualization:
    def __init__(self):
        pass

    # Matplotlib config
    def visualize_images(self):
        plt.rc('image', cmap='gray_r')
        plt.rc('grid', linewidth=0)
        plt.rc('xtick', top=False, bottom=False, labelsize='large')
        plt.rc('ytick', left=False, right=False, labelsize='large')
        plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
        plt.rc('text', color='a8151a')
        plt.rc('figure', facecolor='F0F0F0')# Matplotlib fonts


    # utility to display a row of digits with their predictions
    def display_images(self, left, right, predictions, labels, title, n):
        plt.figure(figsize=(17,3))
        plt.title(title)
        plt.yticks([])
        plt.xticks([])
        plt.grid(None)
        left = np.reshape(left, [n, 28, 28])
        left = np.swapaxes(left, 0, 1)
        left = np.reshape(left, [28, 28*n])
        plt.imshow(left)
        plt.figure(figsize=(17,3))
        plt.yticks([])
        plt.xticks([28*x+14 for x in range(n)], predictions)
        for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
            if predictions[i] > 0.5: t.set_color('red') # bad predictions in red
        plt.grid(None)
        right = np.reshape(right, [n, 28, 28])
        right = np.swapaxes(right, 0, 1)
        right = np.reshape(right, [28, 28*n])
        plt.imshow(right)
    
    def plot_metrics(self, history, metric_name, title, ylim=5):
        plt.title(title)
        plt.ylim(0, ylim)
        plt.plot(history.history[metric_name],color='blue',label=metric_name)
        plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return (y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss

if __name__ == '__main__':
    # Loading dataset
    tr_pairs, tr_y, ts_pairs, ts_y = FashionMnistPairsDataset().fashion_mnist_pairs_dataset()

    # Defining Siamese Netowkr
    siamese_model = SiameseNetwork().build_siamese_network()

    # Compiling model and training the model.
    rms = RMSprop()
    siamese_model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
    history = siamese_model.fit([tr_pairs[:,0], tr_pairs[:,1]], tr_y, epochs=2, batch_size=128, validation_data=([ts_pairs[:,0], ts_pairs[:,1]], ts_y))

    # Evaluating the model
    siamese_model.evaluate(x=[ts_pairs[:, 0], ts_pairs[:, 1]], y=ts_y)
    loss = siamese_model.evaluate(x=[ts_pairs[:,0],ts_pairs[:,1]], y=ts_y)

    y_pred_train = siamese_model.predict([tr_pairs[:,0], tr_pairs[:,1]])
    train_accuracy = compute_accuracy(tr_y, y_pred_train)

    y_pred_test = siamese_model.predict([ts_pairs[:,0], ts_pairs[:,1]])
    test_accuracy = compute_accuracy(ts_y, y_pred_test)

    print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))

    # Display results
    y_pred_train = np.squeeze(y_pred_train)
    indexes = np.random.choice(len(y_pred_train), size=10)
    MetricsAndVisualization().display_images(tr_pairs[:, 0][indexes], tr_pairs[:, 1][indexes], y_pred_train[indexes], tr_y[indexes], "clothes and their dissimilarity", 10)