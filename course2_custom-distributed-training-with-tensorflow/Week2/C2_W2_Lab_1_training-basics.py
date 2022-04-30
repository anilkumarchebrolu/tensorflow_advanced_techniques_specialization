'''
Steps for basic training of a network

1. Define a model
2. Prepare the training data
3. Define Loss and optimizer
4. Train the model on training inputs to  minimize loss.
5. validate the model.
'''

import tensorflow as tf
from matplotlib import pyplot as plt

class Model:
    def __init__(self) -> None:
        self.w = tf.Variable(4.0) # trainable weight
        self.b = tf.Variable(3.0) # trainable bias

    def __call__(self, inputs):
        return (self.w * inputs) + self.b # Wx+b

class TrainingData:
    def training_data(self):
        true_w = 11.0
        true_b = -1.0

        num_samples = 1000
        xs = tf.random.normal(shape=[num_samples])
        ys = true_w * xs + true_b
        return (xs, ys)

class MSELoss:
    def __init__(self) -> None:
        pass

    def mse_loss(self, y_pred, y_true):
        return  tf.reduce_mean(tf.square(y_pred - y_true ))

class Train:
    def train(model, inputs, outputs, learning_rate):
        with tf.GradientTape() as tape:
            logits = model(inputs) # forward pass
            loss = MSELoss().mse_loss(logits, outputs) # calculating loss
        
        # calculating the gradients
        dw, db = tape.gradient(loss, [model.w, model.b])

        # updating the weights and biases
        model.w.assign_sub(learning_rate * dw)
        model.b.assign_sub(learning_rate * db)
        return loss

class PlotUtils:
    def __init__(self) -> None:
        pass

    def plot_data(self, inputs, outputs, predicted_outputs, file_name):
        real = plt.scatter(inputs, outputs, c='b', marker='.')
        predicted = plt.scatter(inputs, predicted_outputs, c='r', marker='+')
        plt.legend((real,predicted), ('Real Data', 'Predicted Data'))
        plt.savefig(f"Week2/outputs/{file_name}")
        plt.show()

    def plot_loss_for_weights(self, weights_list, losses):
        for idx, weights in enumerate(weights_list):
            plt.subplot(120 + idx + 1)
            plt.plot(weights['values'], losses, 'r')
            plt.plot(weights['values'], losses, 'bo')
            plt.xlabel(weights['name'])
            plt.ylabel('Loss')
            plt.savefig("Week2/outputs/loss_vs_weights.png")
            

if __name__ == "__main__":
    epochs = 15
    model = Model()
    xs, ys = TrainingData().training_data()
    PlotUtils().plot_data(xs, ys, model(xs), "intial_data.png")

    list_w, list_b = [], []
    losses = []
    for epoch in range(epochs):
        list_w.append(model.w.numpy())
        list_b.append(model.b.numpy())

        current_loss = Train.train(model, xs, ys, learning_rate=0.1)
        losses.append(current_loss)
        print('Epoch %2d: w=%1.2f b=%1.2f, loss=%2.5f' %(epoch, list_w[-1], list_b[-1], current_loss))

    # Predicting and plotting the training data
    predicted_test_outputs = model(xs)
    PlotUtils().plot_data(xs, ys, predicted_test_outputs, "final_data.png")

    # Plotting the weights
    weights_list = [{ 'name' : "w",
                  'values' : list_w
                },
                {
                  'name' : "b",
                  'values' : list_b
                }]

    PlotUtils().plot_loss_for_weights(weights_list, losses)