import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer

class SimpleDense(Layer):
    def __init__(self, units, activation=None, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(SimpleDense, self).__init__(trainable, name, dtype, dynamic, **kwargs)
        self.units = units

        # define the activation
        self.activation = activations.get(activation)

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()

        self.w = tf.Variable(
            name="weights",
            initial_value=w_init(
                shape=(input_shape[-1], self.units),
                dtype='float32'
            ),
            trainable=True
        )

        self.b = tf.Variable(
            name = "bias",
            initial_value=b_init(
                shape=(self.units,),
                dtype='float32'
            ),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)
        
if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        SimpleDense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)