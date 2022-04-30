##  One device Strategy. 
# This is typically used to deliberately test your code on a single device. 
# This can be used before switching to a different strategy that distributes across multiple devices
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


# resize the image and normalize pixel values
def format_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return  image, label


def build_and_compile_model():
    do_fine_tuning = False
    print("Building model with", MODULE_HANDLE)

    # configures the feature extractor fetched from TF Hub
    feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                   input_shape=IMAGE_SIZE + (3,), 
                                   trainable=do_fine_tuning)

    # define the model
    model = tf.keras.Sequential([
      feature_extractor,
      # append a dense with softmax for the number of classes
      tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # display summary
    model.summary()

    # configure the optimizer, loss and metrics
    optimizer = tf.keras.optimizers.SGD(lr=0.002, momentum=0.9) if do_fine_tuning else 'adam'
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

if __name__ == '__main__':
    ## 1. Defining the distribution strategy
    # Checking available GPU devices
    devices = tf.config.list_physical_devices('GPU')
    print(devices[0])

    gpu_name = 'GPU:0'

    # define strategy
    one_strategy = tf.distribute.OneDeviceStrategy(device=gpu_name)

    ## Parameters
    pixels = 224
    MODULE_HANDLE = 'https://tfhub.dev/tensorflow/resnet_50/feature_vector/1'
    IMAGE_SIZE = (pixels, pixels)
    BATCH_SIZE = 128

    print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

    # Downloading and preparing the data
    splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']

    (train_examples, validation_examples, test_examples), info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split=splits)

    num_examples = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes

    # prepare batches
    train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
    validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
    test_batches = test_examples.map(format_image).batch(1)

    # check if the batches have the correct size and the images have the correct shape
    for image_batch, label_batch in train_batches.take(1):
        pass

    print(image_batch.shape)

    # build and compile under the strategy scope
    with one_strategy.scope():
        model = build_and_compile_model()

    EPOCHS = 3
    hist = model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)

    # Once everything is working good. We can switch to different strategy with multiple devices.