import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from matplotlib import pyplot as plt

def generate_data(m):
    '''plots m random points on a 3D plane'''

    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty((m, 3))
    data[:,0] = np.cos(angles) + np.sin(angles)/2 + 0.1 * np.random.randn(m)/2
    data[:,1] = np.sin(angles) * 0.7 + 0.1 * np.random.randn(m) / 2
    data[:,2] = data[:, 0] * 0.1 + data[:, 1] * 0.3 + 0.1 * np.random.randn(m)
    
    return data

if __name__ == '__main__':
    # use the function above to generate data points
    X_train = generate_data(100)
    X_train = X_train - X_train.mean(axis=0, keepdims=0)

    # preview the data
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], cmap='Reds');

    # Building the model
    encoder = Sequential(Dense(2, input_shape=[3]))
    decoder = Sequential(Dense(3, input_shape=[2]))

    autoencoder = Sequential([encoder, decoder])

    # compiling the model
    autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.1))

    # Training the model
    history = autoencoder.fit(X_train, X_train, epochs=200) # as input and output are same = x_train and y_train are same.

    # Ploting the encoder output
    # encode the data
    codings = encoder.predict(X_train)

    # see a sample input-encoder output pair
    print(f'input point: {X_train[0]}')
    print(f'encoded point: {codings[0]}')

    # plot all encoder outputs
    fig = plt.figure(figsize=(4,3))
    plt.plot(codings[:,0], codings[:, 1], "b.")
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
    plt.show()

    # Ploting the decoder output
    # decode the encoder output
    decodings = decoder.predict(codings)

    # see a sample output for a single point
    print(f'input point: {X_train[0]}')
    print(f'encoded point: {codings[0]}')
    print(f'decoded point: {decodings[0]}')

    # plot the decoder output
    ax = plt.axes(projection='3d')
    ax.scatter3D(decodings[:, 0], decodings[:, 1], decodings[:, 2], c=decodings[:, 0], cmap='Reds');
    print("End")