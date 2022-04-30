import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Layer

class IdentityResNetBlock(tf.keras.Model):
    '''
    Defining Identity ResNet Block as a building block for constructing the ResNet Architecture.
    '''
    def __init__(self, filters, kernel_size):
        super(IdentityResNetBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()

        self.act = ReLU()
        self.add = Add()


    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(input_tensor)
        x = self.bn2(x)

        x = self.add([input_tensor, x])
        x = self.act(x)
        return x


class ResNet(tf.keras.Model):
    def __init__(self, num_classes) -> None:
        super(ResNet, self).__init__()
        self.conv1 = Conv2D(64, (7, 7), padding='same')
        self.bn1 = BatchNormalization()
        self.pool = MaxPool2D()
        self.act = ReLU()

        self.res_block_1 = IdentityResNetBlock(64, 3)
        self.res_block_2 = IdentityResNetBlock(64, 3)

        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)

        x = self.global_avg_pool(x)
        return self.dense(x)


def preprocess(features):
    return tf.cast(features['image'], tf.float32)/255., features['label']

if __name__ == '__main__':
    # Building Resnet model for MNIST
    resnet = ResNet(10)
    resnet.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # load and preprocess the dataset
    dataset = tfds.load('mnist', split=tfds.Split.TRAIN, data_dir="raw_data")
    dataset = dataset.map(preprocess).batch(32)

    # Train the model
    resnet.fit(dataset, epochs=5)