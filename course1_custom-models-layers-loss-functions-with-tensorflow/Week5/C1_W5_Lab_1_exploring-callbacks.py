import tensorflow as tf
import tensorflow_datasets as tfds
import datetime
import os
import math
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler, ReduceLROnPlateau

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

class BuildModel:
    '''
    Building a simple model
    '''
    def build_model(dense_units, input_shape=IMAGE_SIZE + (3,)):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense_units, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        return model

class HorsesOrHumanDataset:

    def format_image(self, image, label):
        image = tf.image.resize(image, IMAGE_SIZE) / 255.0
        return  image, label

    def horses_or_human_dataset(self):
        path = "raw_data/horses_or_humans"
        splits, info = tfds.load('horses_or_humans', data_dir=path, as_supervised=True, with_info=True, split=['train[:80%]', 'train[80%:]', 'test'])

        (train_examples, validation_examples, test_examples) = splits

        num_examples = info.splits['train'].num_examples
        num_classes = info.features['label'].num_classes

        train_batches = train_examples.shuffle(num_examples // 4).map(self.format_image).batch(BATCH_SIZE).prefetch(1)
        validation_batches = validation_examples.map(self.format_image).batch(BATCH_SIZE).prefetch(1)
        test_batches = test_examples.map(self.format_image).batch(1)
        return train_batches, validation_batches, test_batches

class Model_trainer:
    def model_trainer(call_back):
        model = BuildModel.build_model(dense_units=256)
        model.compile(
            optimizer='sgd',
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])

        train_batches, validation_batches, test_batches = HorsesOrHumanDataset().horses_or_human_dataset()

        model.fit(train_batches, 
                epochs=20, 
                validation_data=validation_batches, 
                callbacks=call_back)



def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 1
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

if __name__ == '__main__':
    # Understanding tensorboard callback
    logdir = os.path.join("Week5/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(logdir)

    Model_trainer.model_trainer([tensorboard_callback])

    # Understanding Model checkout callback
    Model_trainer.model_trainer([ModelCheckpoint('Week5/outputs/weights/weights.{epoch:02d}-{val_loss:.2f}.h5', verbose=1)])
    Model_trainer.model_trainer([ModelCheckpoint('Week5/outputs/saved_model', verbose=1)])
    Model_trainer.model_trainer([ModelCheckpoint('Week5/outputs/model.h5', verbose=1)])

    # Understanding early stopping
    early_stopping = EarlyStopping(patience=3,
                                    min_delta=0.05,
                                    baseline=0.8,
                                    mode='min',
                                    monitor='val_loss',
                                    restore_best_weights=True,
                                    verbose=1)
    Model_trainer.model_trainer([early_stopping])

    # Understanding csv logger
    csv_file = 'Week5/outputs/training.csv'
    Model_trainer.model_trainer([CSVLogger(csv_file)])

    # Understanding Learning Rate Scheduler
    Model_trainer.model_trainer([LearningRateScheduler(step_decay, verbose=1), TensorBoard(log_dir='./log_dir')])

    # Understanding ReduceLROnPlateau
    Model_trainer.model_trainer(
        [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=1, min_lr=0.001),
            TensorBoard(log_dir='./log_dir')
        ]
    )