import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import HyperModel
import kerastuner as kt
import numpy as np
from potential_flow import ME_phi, ME_p, E_p, E_phi, ME_y, PotentialFlowEnv, SensorArray, gather_p, gather_phi
import potential_flow
from sklearn.utils import shuffle as sk_shuffle


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
    # physics_informed = [False, "phi", "u"]
    def __init__(self, pfenv: PotentialFlowEnv, n_layers=1, units=[256],
                 physics_informed_u=False, physics_informed_phi=False,
                 phi_gradient=True, alpha=1, window_size=1, print_summary=True):
        super(MLP, self).__init__()
        self.pfenv = pfenv
        self.s_bar = pfenv.sensor()
        self.alpha = alpha
        self.physics_informed_u = physics_informed_u
        self.physics_informed_phi = physics_informed_phi

        self.p_loss = self._MAE_normalized if phi_gradient else self._MAE_p_normalized

        if window_size == 1:
            input_shape = (2*tf.size(self.s_bar), )
        else:
            input_shape = (window_size, 2*tf.size(self.s_bar))

        # Physics informed only output p = (b, d)
        n_output_units = pfenv.Y_BAR_SIZE if not physics_informed_phi else pfenv.Y_BAR_SIZE - 1

        self.D_2SQ = 1 / (2.*self.pfenv.dimensions[1])
        self.add(layers.Flatten(input_shape=input_shape))
        self.add(RescaleProfile())
        # self.add(CubicRootNormalize(pfenv))
        # Make layer that scales the u_x_bar and u_y_bar separately on perhaps current max* or overal max.

        # self.add(layers.experimental.preprocessing.Rescaling(scale=scale, input_shape=input_shape))

        for i in range(n_layers):
            # Different activation functions.
            self.add(layers.Dense(units[i], activation="relu"))

        self.add(layers.Dense(n_output_units))
        # Either transform and wrap phi. Or use special loss function.

        if print_summary:
            self.summary()

    def call(self, inputs, training=False, mask=None):
        outputs = super().call(inputs, training=training, mask=mask)
        if self.physics_informed_phi:
            phi = self._phi_calc(inputs, outputs)
            outputs = tf.concat([outputs, tf.reshape(phi, [-1, 1])], axis=-1)
        return outputs

    def compile(self, learning_rate=1e-3):
        super(MLP, self).compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate),  # Optimizer
            # Loss function to minimize
            # MSE_y
            loss=self.p_loss,
            # List of metrics to monitor
            metrics=[potential_flow.ME_p, potential_flow.ME_phi,
                     potential_flow.ME_y(self.pfenv)],
            run_eagerly=False
        )

    def train_step(self, data):
        u, y = data

        with tf.GradientTape() as tape:
            y_pred = self(u, training=True)  # Forward pass of the MLP
            # y_pred will only yield b and d when PINN mode.

            # Compute the loss values
            # (the loss function is configured in `compile()`)
            loss_y = self.compiled_loss(y, y_pred)
            # tf.print(loss_y)
            if self.physics_informed_u:
                # Forward pass of the potential flow model.
                u_pred = self.pfenv(y_pred)
                u_true = self.pfenv(y)

                loss_u = self._PINN_MSE(u_true, u_pred)
                # Summing losses for single gradient.
                # tf.print(loss_u)
                loss = loss_u
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

    def _phi_calc(self, u, p):

        def phi_calc_inner(u_p_pair):
            u, p = u_p_pair
            u_xy = tf.split(u, 2, axis=-1)
            rho = self.pfenv.rho(self.s_bar, p[0], p[1])

            Psi_e = self.pfenv.Psi_e(rho)
            Psi_n = self.pfenv.Psi_n(rho)
            Psi_o = self.pfenv.Psi_o(rho)

            cos_phi = tf.reduce_mean(u_xy[0] * Psi_n - u_xy[1] * Psi_o)
            sin_phi = tf.reduce_mean(u_xy[1] * Psi_e - u_xy[0] * Psi_o)

            phi = tf.constant(np.pi) + tf.atan2(sin_phi, cos_phi)
            return phi

        return tf.vectorized_map(phi_calc_inner, (u, p))

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

    def _MSE_p_normalized(self, y_true, y_pred):
        # tf.shape for dynamic shape of Tensor
        N = tf.cast(tf.shape(y_true)[0], tf.float32)
        MSE_p = 2 * self.D_2SQ ** 2 * \
            tf.reduce_sum(tf.square(gather_p(y_true) - gather_p(y_pred)))
        MSE_out = MSE_p / (2. * N)
        return MSE_out

    def _MAE_p_normalized(self, y_true, y_pred):
        # tf.shape for dynamic shape of Tensor
        N = tf.cast(tf.shape(y_true)[0], tf.float32)
        AE_p = self.D_2SQ * tf.reduce_sum(E_p(y_true, y_pred, ord=1))
        MAE_out = AE_p / N
        return MAE_out

    def _MAE_normalized(self, y_true, y_pred):
        # tf.shape for dynamic shape of Tensor
        N = tf.cast(tf.shape(y_true)[0], tf.float32)
        AE_p = self.D_2SQ * tf.reduce_sum(E_p(y_true, y_pred, ord=1))
        AE_phi = tf.reduce_sum(E_phi(y_true, y_pred)) / np.pi
        MAE_out = (AE_p + AE_phi) / (2. * N)
        return MAE_out

    def _MAE_2_normalized(self, y_true, y_pred):
        # tf.shape for dynamic shape of Tensor
        N = tf.cast(tf.shape(y_true)[0], tf.float32)
        MAE_p = self.D_2SQ * \
            tf.reduce_sum(tf.abs(gather_p(y_true) - gather_p(y_pred)))
        MAE_phi = tf.reduce_sum(
            tf.abs(tf.cos(gather_phi(y_true)) - tf.cos(gather_phi(y_pred))))
        MAE_phi += tf.reduce_sum(
            tf.abs(tf.sin(gather_phi(y_true)) - tf.sin(gather_phi(y_pred))))

        MAE_out = (MAE_p + MAE_phi) / (4. * N)
        return MAE_out

    def _PINN_MSE(self, u_true, u_pred):
        def _rescale(x):
            abs_max = tf.reduce_max(tf.abs(x), axis=1)
            output = x/tf.reshape(abs_max, (-1, 1))
            return output

        def rescale_profile(inputs):
            u_x, u_y = tf.split(inputs, 2, axis=1)
            u_x = _rescale(u_x)
            u_y = _rescale(u_y)
            outputs = tf.concat([u_x, u_y], 1)
            return outputs

        u_true = rescale_profile(u_true)
        u_pred = rescale_profile(u_pred)

        return keras.losses.MAE(u_true, u_pred)

    def evaluate_full(self, samples_u, samples_y):
        pred_y = self.predict(samples_u)
        pred_y = tf.convert_to_tensor(pred_y, dtype=tf.float32)
        true_y = tf.convert_to_tensor(samples_y, dtype=tf.float32)
        p_eval = E_p(true_y, pred_y)
        phi_eval = E_phi(true_y, pred_y)
        print(ME_p(true_y, pred_y))
        print(ME_phi(true_y, pred_y))
        print(ME_y(self.pfenv)(true_y, pred_y))
        return p_eval, phi_eval


class MLPHyperModel(HyperModel):
    def __init__(self, pfenv, physics_informed_phi=False, phi_gradient=True, window_size=1):
        self.pfenv = pfenv
        self.physics_informed_phi = physics_informed_phi
        self.phi_gradient = phi_gradient
        self.window_size = window_size

    def build(self, hp):
        n_layers = hp.Int("num_hidden_layers", min_value=1,
                          max_value=5, default=1)
        units = []
        for i in range(n_layers):
            units.append(hp.Int("units_" + str(i),
                                min_value=32, max_value=2048, step=1, default=32))
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling='log')

        mlp = MLP(self.pfenv, n_layers, units, physics_informed_phi=self.physics_informed_phi,
                  phi_gradient=self.phi_gradient, window_size=self.window_size)
        mlp.compile(learning_rate)

        return mlp


class MLPTuner(kt.tuners.BayesianOptimization):
    def run_trial(self, trial, *fit_args, **fit_kwargs):
        hp = trial.hyperparameters
        batch_size = 2048
        # hp.Int("batch_size", min_value=32,
        #                     max_value=2048, step=32, default=32)
        fit_kwargs['batch_size'] = batch_size
        fit_kwargs['callbacks'] = [
            tf.keras.callbacks.EarlyStopping('val_ME_y', patience=5)]

        super(MLPTuner, self).run_trial(trial, *fit_args, **fit_kwargs)


def main():
    m = MLP(PotentialFlowEnv(sensor=SensorArray(8)))

    y_true = tf.constant([[1., 1., 3.5]])
    y_pred = tf.constant([[2., 3., 3.4]])

    print(potential_flow.ME_p(y_true, y_pred))
    print(potential_flow.ME_phi(y_true, y_pred))
    print(potential_flow.ME_phi(y_true, y_pred))

    print(potential_flow.MSE(y_true, y_pred))
    print(m._MAE_normalized(y_true, y_pred))


if __name__ == "__main__":
    main()
