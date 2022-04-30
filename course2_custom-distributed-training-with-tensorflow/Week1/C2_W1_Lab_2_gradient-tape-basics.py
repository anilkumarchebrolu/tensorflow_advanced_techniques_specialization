import tensorflow as tf

class GradientTapeBasicExecise:
    def __init__(self) -> None:
        pass

    def gradient_tape_basic_exercise(self):
        x = tf.ones((2, 2))

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = tf.reduce_sum(x)
            z = tf.square(y)
        
        dz_dx = tape.gradient(z, x)
        print(f"gradient of basic exercise {dz_dx}")

class GradientTapePersistentExecise:
    def __init__(self) -> None:
        pass

    def gradient_tape_persistent_exercise(self):
        x = tf.ones((2, 2))

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = tf.reduce_sum(x)
            z = tf.square(y)
        
        dz_dx = tape.gradient(z, x)
        dy_dx = tape.gradient(y, x)
        del tape
        print(f"gradient of dz/dx exercise {dz_dx}")
        print(f"gradient of dy/dx exercise {dy_dx}")

class GradientTapeNested:
    def __init__(self) -> None:
        pass

    def gradient_tape_nested():
        x = tf.Variable(1.0)

        with tf.GradientTape() as tape_1:
            with tf.GradientTape() as tape_2:
                y = x * x * x
            
            dy_dx = tape_2.gradient(y, x) # 3 x^2 
        d2y_dx2 = tape_1.gradient(dy_dx, x) # 6x

        print(f"gradient of dy_dx exercise {dy_dx}")
        print(f"gradient of d2y_dx2 exercise {d2y_dx2}")



if __name__ == '__main__':
    GradientTapeBasicExecise().gradient_tape_basic_exercise()
    GradientTapePersistentExecise().gradient_tape_persistent_exercise()
    GradientTapeNested.gradient_tape_nested()

        