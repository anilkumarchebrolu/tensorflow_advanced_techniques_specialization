'''

'''

# Imports
import pandas as pd
import numpy as np
from scipy.sparse import data
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt

class Plots:
    def __init__(self):
        pass

    def plot_diff(self, y_true, y_pred, title=''):
        plt.scatter(y_true, y_pred)
        plt.title(title)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plt.plot([-100, 100], [-100, 100])
        plt.show()


    def plot_metrics(self, metric_name, title, ylim=5):
        plt.title(title)
        plt.ylim(0, ylim)
        plt.plot(history.history[metric_name], color='blue', label=metric_name)
        plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
        plt.show()

class EnergyEfficiencyDataPreparation:
    '''
    Source: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency

    About Dataset
        X1 Relative Compactness
        X2 Surface Area
        X3 Wall Area
        X4 Roof Area
        X5 Overall Height
        X6 Orientation
        X7 Glazing Area
        X8 Glazing Area Distribution
        y1 Heating Load
        y2 Cooling Load
    '''
    def __init__(self) -> None:
        # self.energy_efficiency_dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        self.energy_efficiency_dataset_excel = "energy_efficiency_data_set\ENB2012_data.xlsx"

    def format_output(self, data):
        y1 = data.pop('Y1')
        y1 = np.array(y1)

        y2 = data.pop('Y2')
        y2 = np.array(y2)
        return y1, y2


    def norm(self, x, train_stats):
        return (x - train_stats['mean'])/train_stats['std']
    
    def loading_energy_efficiency_dataset(self):
        '''
        Loads energy efficiency dataset
        1. Reads data from url into dataframe.
        2. splits data from train and test.
        3. Format and normalize train and test data.
        '''
        # Loading energy efficiency dataset
        dataframe = pd.read_excel(self.energy_efficiency_dataset_excel, engine='openpyxl')
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        # Removing all the columns and rows with nan values.
        dataframe = dataframe.dropna(axis=0, how='all')
        dataframe = dataframe.dropna(axis=1, how='all')

        # Split data into train and test splits 80/20
        train, test = train_test_split(dataframe, test_size=0.2)
        train_stats = train.describe()

        # Get y1 and y2 as two outputs and format them as numpy arrays
        train_stats.pop('Y1')
        train_stats.pop('Y2')
        train_stats = train_stats.transpose()
        train_y = self.format_output(train)
        test_y = self.format_output(test)

        # Normalize the train and test data
        norm_train_X = self.norm(train, train_stats)
        norm_test_X = self.norm(test, train_stats)

        return norm_train_X, norm_test_X, train_y, test_y


class BuildMultiOutputModel:
    def __init__(self, input_dim) -> None:
        self.input_dim = input_dim

    def model(self):
        # Model definition
        input = Input(shape=(self.input_dim,))
        first_dense = Dense(128, activation='relu')(input)
        second_dense = Dense(128, activation='relu')(first_dense)

        # First split
        y1_output = Dense(1, name='y1_output')(second_dense)

        # Second split
        third_dense = Dense(64, activation='relu')(second_dense)
        y2_output = Dense(1, name='y2_output')(third_dense)

        # Defining the model
        model = Model(inputs=input, outputs= [y1_output, y2_output])
        print(model.summary())
        return model

if __name__ == '__main__':
    
    # Loading data
    dataset = EnergyEfficiencyDataPreparation()
    norm_train_X, norm_test_X, train_y, test_y = dataset.loading_energy_efficiency_dataset()

    # Loading model
    model = BuildMultiOutputModel(input_dim=8).model()

    # Compiling the model.
    model.compile(
        optimizer = tf.keras.optimizers.SGD(lr= 0.001),
        loss = {
            'y1_output': 'mse',
            'y2_output': 'mse'
        },
        metrics={
            'y1_output': tf.keras.metrics.RootMeanSquaredError(),
            'y2_output': tf.keras.metrics.RootMeanSquaredError()
        }
    )

    # Train the model for 500 epochs
    history = model.fit(norm_train_X, train_y, epochs=50, batch_size=10, validation_data=(norm_test_X, test_y))

    ## Model Evaluation
    # Test the model and print loss and mse for both outputs
    loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X, y=test_y)
    print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))

    # Plot the loss and mse
    Y_pred = model.predict(norm_test_X)
    plots = Plots()
    plots.plot_diff(y_true=test_y[0], y_pred=Y_pred[0], title='Y1')
    plots.plot_diff(y_true=test_y[1], y_pred=Y_pred[1], title='Y2')
    plots.plot_metrics(metric_name='y1_output_root_mean_squared_error', title='Y1 RMSE', ylim=6)
    plots.plot_metrics(metric_name='y2_output_root_mean_squared_error', title='Y2 RMSE', ylim=7)
    