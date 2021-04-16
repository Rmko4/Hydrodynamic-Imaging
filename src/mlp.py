import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from potential_flow import PotentialFlowEnv, SensorArray


class RescaleProfile(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RescaleProfile, self).__init__(**kwargs)

    def _rescale(self, x):
        abs_max = tf.reduce_max(tf.abs(x), axis=1)
        output = x/tf.reshape(abs_max, (-1, 1))
        return output

    def call(self, inputs):
        u_x, u_y = tf.split(inputs, 2, axis=1)
        u_x = self._rescale(u_x)
        u_y = self._rescale(u_y)
        outputs = tf.concat([u_x, u_y], 1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class CubicRootNormalize(keras.layers.Layer):
    def __init__(self, pfenv: PotentialFlowEnv, **kwargs):
        super(CubicRootNormalize, self).__init__(**kwargs)
        self.pfenv = pfenv

    def _cube_root(self, x):
        sign = tf.sign(x)
        x = sign * tf.pow(abs(x), 1./3.)
        return x

    def _normalize(self, x, scale):
        x = self._cube_root(x)
        output = scale * x
        return output

    def call(self, inputs):
        u_x, u_y = tf.split(inputs, 2, axis=1)
        scale = np.math.pow(self.pfenv.y_offset, 3) / \
            (self.pfenv.W * np.math.pow(self.pfenv.a, 3))

        u_x = self._normalize(u_x, 0.5*scale)
        u_y = self._normalize(u_y, scale)

        outputs = tf.concat([u_x, u_y], 1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def MED_p(y_true, y_pred):
    L2_norm = tf.sqrt(tf.reduce_sum(tf.square(y_true[:, 0:2] - y_pred[:, 0:2]), axis=-1))
    return tf.reduce_mean(L2_norm)

def MDE_phi(y_true, y_pred):
    phi_e = y_true[:, 2] - y_pred[:, 2]
    abs_atan2 = tf.abs(tf.atan2(tf.sin(phi_e), tf.cos(phi_e)))
    return tf.reduce_mean(abs_atan2)

def MSE(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


class MLP(keras.Sequential):
    def __init__(self, pfenv: PotentialFlowEnv):
        super(MLP, self).__init__()
        self.pfenv = pfenv
        s_bar = pfenv.getSensor().s_bar
        input_shape = (2*s_bar.size,)
        self.add(layers.InputLayer(input_shape=input_shape))
        self.add(RescaleProfile())
        # self.add(CubicRootNormalize(pfenv))
        # Make layer that scales the u_x_bar and u_y_bar separately on perhaps current max* or overal max.

        # self.add(layers.experimental.preprocessing.Rescaling(scale=scale, input_shape=input_shape))
        self.add(layers.Dense(100, activation="sigmoid"))
        self.add(layers.Dense(pfenv.Y_BAR_SIZE))
        # Either transform and wrap phi. Or use special loss function.
        self.summary()

    def compile(self):
        super(MLP, self).compile(
            optimizer=keras.optimizers.Adam(),  # Optimizer
            # Loss function to minimize
            loss=keras.losses.MeanSquaredError(),
            # List of metrics to monitor
            metrics=[keras.losses.MSE, MSE, MED_p, MDE_phi],
        )

    def _normalize_y(y_bar):
        return tf.constant([y_bar[:, 0], y_bar[:, 1], tf.cos(y_bar[:, 2]), tf.sin(y_bar[:, 2])])

    def _MSE_normalize():
        return


def main():
    # m = MLP(PotentialFlowEnv(sensor=SensorArray(8)))
    y_true = tf.constant([[1., 1., 0.], [2., 3., 3.]])
    y_pred = tf.constant([[2., 3., 6], [2., 3., 3.]])
    # print(rMSE_p(y_true, y_pred))
    # print(MSE_p(y_true, y_pred))
    print(MED_p(y_true, y_pred))


if __name__ == "__main__":
    main()
