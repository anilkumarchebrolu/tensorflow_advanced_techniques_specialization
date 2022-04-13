from matplotlib.pyplot import plot
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

if __name__ == '__main__':
    # Definint Input Layers
    deep_input = Input(shape=[1], name="Deep_Input")
    wide_input = Input(shape=[1], name="Wide_input")

    # Defining Dense
    hidden_1 = Dense(32, activation='relu')(deep_input)
    hidden_2 = Dense(32, activation='relu')(hidden_1)

    # concatenate
    concat = Concatenate()([wide_input, hidden_2])

    # outputs
    aux_output = Dense(1, name="aux_output")(hidden_2)
    output = Dense(1, name="output")(concat)

    # Defining Model
    model = Model(inputs=[deep_input, wide_input], outputs=[aux_output, output])

    print(model.summary())
    
    # Ploting the model
    plot_model(model, to_file="Week4/outputs/function_api_plotting.png")

