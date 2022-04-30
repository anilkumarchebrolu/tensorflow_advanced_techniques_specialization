import tensorflow as tf
import tensorflow_datasets as tfds
import os


# Function for normalizing the image
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64

if __name__ == '__main__':
    # Loading data
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True, data_dir='./data')
    mnist_train, mnist_test = datasets['train'], datasets['test']

    # Define the strategy to use and print the number of devices found
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Use for Mirrored Strategy
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    # Set up the train and eval data set
    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
    # train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # eval_dataset = strategy.experimental_distribute_dataset(eval_dataset)

    # Use for Mirrored Strategy -- comment out `with strategy.scope():` and deindent for no strategy
    with strategy.scope():
        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
        ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

    model.fit(train_dataset, epochs=12)