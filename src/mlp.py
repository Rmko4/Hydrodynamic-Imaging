import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from potential_flow import PotentialFlowEnv, SensorArray


class RescaleProfile(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RescaleProfile, self).__init__(**kwargs)

    def _rescale(self, x):
        return x/tf.reduce_max(tf.abs(x))

    def call(self, inputs):
        u_x, u_y = tf.split(inputs, 2)
        u_x = self._rescale(u_x)
        u_y = self._rescale(u_y)
        outputs = tf.concat([u_x, u_y], 0)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class MLP(keras.Sequential):
    def __init__(self, pfenv: PotentialFlowEnv):
        super(MLP, self).__init__()
        self.pfenv = pfenv
        s_bar = pfenv.getSensor().s_bar
        input_shape = (2*s_bar.size,)
        self.add(layers.InputLayer(input_shape=input_shape))
        self.add(RescaleProfile())
        # Make layer that scales the u_x_bar and u_y_bar separately on perhaps current max* or overal max.

        # scale = np.math.pow(pfenv.y_offset, 3) / (pfenv.W * np.math.pow(pfenv.a, 3))
        # self.add(layers.experimental.preprocessing.Rescaling(scale=scale, input_shape=input_shape))
        self.add(layers.Dense(100, activation="sigmoid"))
        self.add(layers.Dense(pfenv.Y_BAR_SIZE))
        # Either transform and wrap phi. Or use special loss function.
        self.summary()

    def compile(self):
        super(MLP, self).compile(
            optimizer=keras.optimizers.Adam(),  # Optimizer
            # Loss function to minimize
            loss=keras.losses.MeanAbsoluteError(),
            # List of metrics to monitor
            metrics=[keras.metrics.MeanAbsoluteError()]
        )


def main():
    m = MLP(PotentialFlowEnv(sensor=SensorArray(8)))


if __name__ == "__main__":
    main()
