import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker

class Data:
    def __init__(self):
        pass
    
    def format_image(self,data):
        image = data['image']
        image = tf.reshape(image, [-1])
        image = tf.cast(image, dtype='float32')
        image = image/255.0
        return image, data['label']

    def fashion_mnist(self):
        # Loading train and test data
        train_data, info = tfds.load("fashion_mnist", split="train", data_dir="Week2/data", with_info=True, download=True)
        test_data = tfds.load("fashion_mnist", split="test", data_dir="Week2/data", download=False)

        # Shuffle and batch training and test data.
        train_data = train_data.map(self.format_image)
        test_data = test_data.map(self.format_image)

        # batching the dataset
        batch_size = 64
        train_data = train_data.shuffle(buffer_size=1024).batch(batch_size)
        test_data =  test_data.batch(batch_size=batch_size)
        return (train_data, test_data)

class Model:
    def model(self):
        inputs = tf.keras.Input(shape=(784,), name='digits')
        x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

class Train:
    def train_data_for_one_epoch(self, model, train_data, loss_object, optimizer, train_acc_metric):
        losses = []
        pbar = tqdm(total=len(list(enumerate(train_data))), position=0, leave=True)
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)
                loss = loss_object(y_pred=logits, y_true=y_batch_train)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            losses.append(loss)
            train_acc_metric(y_batch_train, logits)
            pbar.set_description(f"Training loss for step {step} is {loss}")
            pbar.update()
        return losses

    def perform_validation(self, model, test_dataset, loss_object, val_acc_metric):
        losses = []
        for x_val, y_val in test_dataset:
            val_logits = model(x_val)
            val_loss = loss_object(y_true=y_val, y_pred=val_logits)
            losses.append(val_loss)
            val_acc_metric(y_val, val_logits)
        return losses

class Plots:
    def plot_metrics(self, train_metric, val_metric, metric_name, title, ylim=5):
        plt.title(title)
        plt.ylim(0,ylim)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.plot(train_metric,color='blue',label=metric_name)
        plt.plot(val_metric,color='green',label='val_' + metric_name)
        plt.savefig("Week2/outputs/metrics.png")


if __name__ == "__main__":
    # Defining the model
    model = Model().model()
    model.summary()

    # Loading the data
    train_data, test_data = Data().fashion_mnist()

    # training
    loss_object = SparseCategoricalCrossentropy()
    train_acc_metric = SparseCategoricalAccuracy()
    val_acc_metric = SparseCategoricalAccuracy()
    adam = Adam()
    
    epochs = 10
    epochs_val_losses, epochs_train_losses = [], []
    train_data_for_one_epoch = Train().train_data_for_one_epoch
    perform_validation = Train().perform_validation


    for epoch in range(epochs):
        print(f"Epoch {epoch} started")
        train_loss = train_data_for_one_epoch(model, train_data, loss_object, adam, train_acc_metric)
        train_acc = train_acc_metric.result()

        test_loss = perform_validation(model, test_data, loss_object, val_acc_metric)
        test_acc = val_acc_metric.result()

        losses_train_mean = np.mean(train_loss)
        losses_val_mean = np.mean(test_loss)
        epochs_val_losses.append(losses_val_mean)
        epochs_train_losses.append(losses_train_mean)

        print(f"Train loss: {losses_train_mean} Test loss: {losses_val_mean} Train accuracy: {train_acc} Test accuracy: {test_acc}")

        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
    
    Plots().plot_metrics(epochs_train_losses, epochs_val_losses, "Loss", "Loss", ylim=1.0)

    test_accuracy = model.evaluate(test_data)
    print(test_accuracy)


        


