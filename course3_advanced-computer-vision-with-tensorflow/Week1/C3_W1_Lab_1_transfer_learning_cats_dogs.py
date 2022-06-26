import urllib.request
import zipfile
import os
import random
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


CAT_SOURCE_DIR =  "data/cats_vs_dogs/PetImages/Cat"
DOG_SOURCE_DIR = "data/cats_vs_dogs/PetImages/Dog"

TRAINING_CATS_DIR = "data/cats_vs_dogs/training/cats"
TESTING_CATS_DIR = "data/cats_vs_dogs/testing/cats"

TRAINING_DOGS_DIR = "data/cats_vs_dogs/training/dogs"
TESTING_DOGS_DIR = "data/cats_vs_dogs/testing/dogs"

TRAINING_DIR = "data/cats_vs_dogs/training"
TESTING_DIR = "data/cats_vs_dogs/testing"


class CatsVsDogsData:
    def __init__(self) -> None:
        pass

    def cats_vs_dogs_data(self, data_dir="data/cats_vs_dogs", data_file_name = "data/catsdogs.zip"):
        data_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
        urllib.request.urlretrieve(data_url, data_file_name)
        zip_ref = zipfile.ZipFile(data_file_name, 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        print("Number of cat images:",len(os.listdir(f'{data_dir}/PetImages/Cat/')))
        print("Number of dog images:", len(os.listdir(f'{data_dir}/PetImages/Dog/')))


class SplitData:
    def __init__(self) -> None:
        pass

    def split_data(self, SOURCE, TRAINING, TESTING, SPLIT_SIZE):
        files = []
        for filename in os.listdir(SOURCE):
            file = os.path.join(SOURCE, filename)
            if os.path.getsize(file) > 0:
                files.append(filename)
            else:
                print(filename + " is zero length, so ignoring.")

        training_length = int(len(files) * SPLIT_SIZE)
        testing_length = int(len(files) - training_length)
        shuffled_set = random.sample(files, len(files))
        training_set = shuffled_set[0:training_length]
        testing_set = shuffled_set[training_length:]

        for filename in training_set:
            this_file = os.path.join(SOURCE, filename)
            destination = os.path.join(TRAINING, filename)
            copyfile(this_file, destination)

        for filename in testing_set:
            this_file = os.path.join(SOURCE, filename)
            destination = os.path.join(TESTING, filename)
            copyfile(this_file, destination)


class InceptionCustomModel:
    def __init__(self) -> None:
        pass

    def model(self):
        inception_v3 = InceptionV3(include_top=False, input_shape=(150, 150, 3))
        for layer in inception_v3.layers:
            layer.trainable = False

        last_layer_output = inception_v3.get_layer("mixed10").output

        # Flatten the output layer to 1 dimension
        x = layers.Flatten()(last_layer_output)
        # Add a fully connected layer with 1,024 hidden units and ReLU activation
        x = layers.Dense(1024, activation='relu')(x)
        # Add a final sigmoid layer for classification
        x = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inception_v3.input, x)
        return model



if __name__ == '__main__':
    # Loading cats and dog data
    # CatsVsDogsData().cats_vs_dogs_data()
    
    # Split data into training and testing
    split_size = .9
    SplitData().split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    SplitData().split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

    print("Number of training cat images", len(os.listdir(TRAINING_CATS_DIR)))
    print("Number of training dog images", len(os.listdir(TRAINING_DOGS_DIR)))
    print("Number of testing cat images", len(os.listdir(TESTING_CATS_DIR)))
    print("Number of testing dog images", len(os.listdir(TESTING_DOGS_DIR)))

    # Data Augmentation and loading
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=100,
        class_mode='binary',
        target_size=(150, 150)
    )

    valid_datagen = ImageDataGenerator(
        rescale=1./255
    )

    valid_generator = valid_datagen.flow_from_directory(
        TESTING_DIR,
        batch_size=100,
        class_mode='binary',
        target_size=(150, 150)
    )

    # Create the model
    inception_custom_model = InceptionCustomModel().model()

    inception_custom_model.compile(
        optimizer = RMSprop(lr = 0.001),
        loss = "binary_crossentropy",
        metrics = ['accuracy']
    )

    inception_custom_model.fit(
        train_generator,
        validation_data = valid_generator,
        epochs=2,
        verbose=1
    )