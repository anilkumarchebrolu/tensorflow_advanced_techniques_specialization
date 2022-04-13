import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tf2onnx
import onnx

class WideAndDeepModel(Model):
    def __init__(self, *args, **kwargs):
        super(WideAndDeepModel, self).__init__(*args, **kwargs)
        '''initializes the instance attributes'''
        self.hidden_1 = Dense(32, activation='relu')
        self.hidden_2 = Dense(32, activation='relu')
        self.concatenate = Concatenate()
        self.aux_output = Dense(1, name="aux_output")
        self.outputs = Dense(1, name="output")
        

    def call(self, inputs):
        wide_input, deep_input = inputs
        hidden_1 = self.hidden_1(deep_input)
        hidden_2 = self.hidden_2(hidden_1)
        concat_out = self.concatenate([wide_input, hidden_2])
        aux_out = self.aux_output(hidden_2)
        outputs = self.outputs(concat_out)
        return outputs, aux_out

if __name__ == "__main__":
    wide_and_deep_model = WideAndDeepModel()
    wide_and_deep_model([tf.random.normal(shape=[1, 1]), tf.random.normal(shape=[1, 1])])
    plot_model(wide_and_deep_model, expand_nested=True, show_shapes=True, to_file="Week4/outputs/subclassing_api_plotting.png")
    wide_and_deep_model.summary()
    wide_and_deep_model.save("Week4/outputs/wide_and_deep_model", save_format=tf)

    ## To visualize the model storing it as onnx and visualizing the model
    # Storing the model as onnx
    input_signature = [tf.TensorSpec([1, 1], tf.float32, name='x'), tf.TensorSpec([1, 1], tf.float32, name='x2')]
    
    # Use from_function for tf functions
    onnx_model, _ = tf2onnx.convert.from_keras(wide_and_deep_model, input_signature, opset=13)
    onnx.save(onnx_model, "Week4/outputs/wide_and_deep_model/converted_wide_and_deep.onnx")

    