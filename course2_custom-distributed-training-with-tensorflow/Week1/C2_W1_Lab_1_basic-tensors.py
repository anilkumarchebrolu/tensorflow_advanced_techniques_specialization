import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    ## Exercises of basic tensors
    # Create a 1D uint8 NumPy array comprising of first 25 natural numbers
    x = np.arange(0, 25)
    print(x)

    # Convert NumPy array to Tensor using `tf.constant`
    x = tf.constant(x)
    print(x)

    # Square the input tensor x
    x = tf.square(x)
    print(f"Squared x {x}")

    # Reshape tensor x into a 5 x 5 matrix. 
    x = tf.reshape(x, shape=(5, 5))
    print(f"Reshaped x {x}")


    # Try this and look at the error
    # Try to change the input to `shape` to avoid an error
    # x = tf.reshape(x, shape=(2, 5))
    # print("Reshaped x {x}")

    # Cast tensor x into float32. Notice the change in the dtype.
    x = tf.cast(x, dtype=tf.float32)
    print(f"Casted it into float {x}")
    
    # Let's define a constant and see how broadcasting works in the following cell.
    y = tf.constant([2], dtype=tf.float32)

    # Multiply tensor `x` and `y`. `y` is multiplied to each element of x.
    result = tf.multiply(x, y)
    print(f"multiplication {result}")

    # Now let's define an array that matches the number of row elements in the `x` array.
    y = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
    result = x +y
    print(f"addition {result}")