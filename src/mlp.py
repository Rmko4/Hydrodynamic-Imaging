import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from potential_flow import PotentialFlowEnv, SensorArray, gather_p, gather_phi
import potential_flow

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

class MLP(keras.Sequential):

    def __init__(self, pfenv: PotentialFlowEnv):
        super(MLP, self).__init__()
        self.pfenv = pfenv
        s_bar = pfenv.sensor()
        input_shape = (2*tf.size(s_bar),)
        self.D_2SQ = 1/self.pfenv.dimensions[1]
        self.add(layers.InputLayer(input_shape=input_shape))
        self.add(RescaleProfile())
        # self.add(CubicRootNormalize(pfenv))
        # Make layer that scales the u_x_bar and u_y_bar separately on perhaps current max* or overal max.

        # self.add(layers.experimental.preprocessing.Rescaling(scale=scale, input_shape=input_shape))
        self.add(layers.Dense(500, activation="sigmoid"))
        self.add(layers.Dense(pfenv.Y_BAR_SIZE))
        # Either transform and wrap phi. Or use special loss function.
        self.summary()

    def compile(self, physics_informed=False, alpha=1):
        self.alpha = alpha
        self.physics_informed = physics_informed
        self.PINN_loss = keras.losses.MeanSquaredError()

        super(MLP, self).compile(
            optimizer=keras.optimizers.Adam(),  # Optimizer
            # Loss function to minimize
            # MSE_y
            loss=self._MSE_normalized,
            # List of metrics to monitor
            metrics=[potential_flow.ME_p, potential_flow.ME_phi, potential_flow.ME_y(self.pfenv)],
            run_eagerly=False
        )

    def train_step(self, data):
        u, y = data

        with tf.GradientTape() as tape:
            y_pred = self(u, training=True)  # Forward pass of the MLP
            # Compute the loss values
            # (the loss function is configured in `compile()`)
            loss_y = self.compiled_loss(y, y_pred)

            if self.physics_informed:
                # Forward pass of the potential flow model.
                u_pred = self.pfenv(y_pred)
                loss_u = self.PINN_loss(u, u_pred)
                # Summing losses for single gradient.
                loss = (loss_y + self.alpha * loss_u) / (1 + self.alpha)
            else:
                loss = loss_y

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def _MSE_normalized(self, y_true, y_pred):
        # tf.shape for dynamic shape of Tensor
        N = tf.cast(tf.shape(y_true)[0], tf.float32)
        MSE_p = self.D_2SQ * \
            tf.reduce_sum(tf.square(gather_p(y_true) - gather_p(y_pred)))
        MSE_phi = tf.reduce_sum(
            tf.square(tf.cos(gather_phi(y_true)) - tf.cos(gather_phi(y_pred))))
        MSE_phi += tf.reduce_sum(
            tf.square(tf.sin(gather_phi(y_true)) - tf.sin(gather_phi(y_pred))))

        MSE_out = (MSE_p + MSE_phi) / (4. * N)
        return MSE_out

def main():
    m = MLP(PotentialFlowEnv(sensor=SensorArray(8)))

    y_true = tf.constant([[1., 1., 3], [2., 3., 3.]])
    y_pred = tf.constant([[2., 3., 0.], [2., 3., 3.]])

    print(potential_flow.ME_p(y_true, y_pred))
    print(potential_flow.ME_phi(y_true, y_pred))
    print(potential_flow.ME_phi(y_true, y_pred))

    print(potential_flow.MSE(y_true, y_pred))
    print(m._MSE_normalized(y_true, y_pred))


if __name__ == "__main__":
    main()
