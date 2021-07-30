import kerastuner as kt
import numpy as np
import tensorflow as tf
from kerastuner import HyperModel
from sklearn.utils import shuffle as sk_shuffle
from tensorflow import keras
from tensorflow.keras import layers

import potential_flow
from potential_flow import (E_p, E_phi, ME_p, ME_phi, ME_y, PotentialFlowEnv,
                            SensorArray, gather_p, gather_phi)


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
                 pi_u=False, pi_phi=False, phi_gradient=True, pi_learning_rate = 1e-4,
                 pi_clipnorm=1e2, alpha=1, window_size=1, print_summary=True):
        super(MLP, self).__init__()
        self.pfenv = pfenv
        self.s_bar = pfenv.sensor()

        self.alpha = alpha
        self.pi_u = pi_u
        self.pi_phi = pi_phi
        self.pi_learning_rate = pi_learning_rate
        self.pi_clipnorm = pi_clipnorm

        self.y_loss = self._MAE_normalized if phi_gradient else self._MAE_p_normalized
        self.pi_run = False


        if window_size == 1:
            input_shape = (2*tf.size(self.s_bar), )
        else:
            input_shape = (window_size, 2*tf.size(self.s_bar))

        # Physics informed only output p = (b, d)
        n_output_units = pfenv.Y_BAR_SIZE if not pi_phi else pfenv.Y_BAR_SIZE - 1

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
        if self.pi_phi and self.pi_run:
            phi = self._phi_calc(inputs, outputs)
            outputs = tf.concat([outputs, tf.reshape(phi, [-1, 1])], axis=-1)
        return outputs

    def compile(self, learning_rate=1e-3, clipnorm=None):
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                      initial_learning_rate=learning_rate,
                      decay_steps=250,
                      decay_rate=0.9)
        if clipnorm is not None:
            optimizer=keras.optimizers.Adam(lr_schedule, clipnorm=clipnorm)
        else:
            optimizer=keras.optimizers.Adam(lr_schedule)
        super(MLP, self).compile(
            optimizer=optimizer,  # Optimizer
            # Loss function to minimize
            # MSE_y
            loss=self.y_loss,
            # List of metrics to monitor
            metrics=[potential_flow.ME_p, potential_flow.ME_phi,
                     potential_flow.ME_y(self.pfenv)],
            run_eagerly=False
        )
        self.learning_rate = learning_rate



    def train_step(self, data):
        u, y = data

        with tf.GradientTape() as tape:
            y_pred = self(u, training=True)  # Forward pass of the MLP
            # y_pred will only yield b and d when PINN mode.

            # Compute the loss values
            # (the loss function is configured in `compile()`)
            loss_y = self.compiled_loss(y, y_pred)
            # tf.print(loss_y)
            if self.pi_u and self.pi_run:
                # Forward pass of the potential flow model.
                u_pred = self.pfenv(y_pred)
                u_true = self.pfenv(y)

                loss_u = self._PINN_MAE(u, u_pred)
                # Summing losses for single gradient.
                # tf.print(loss_u)
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


    def fit(self,  
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
            
    
        if self.pi_u or self.pi_phi:
            super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs,verbose=verbose, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
            self.pi_run = True
            self.compile(self.pi_learning_rate, self.pi_clipnorm)
            return super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs,verbose=verbose, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
        else:
            return super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs,verbose=verbose, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_batch_size=validation_batch_size, validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)


        

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

    def _PINN_MAE(self, u_true, u_pred):
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
    def __init__(self, pfenv, pi_u=False, pi_phi=False, n_layers=1, units=[32], learning_rate=1e4):
        self.pfenv = pfenv
        self.pi_phi = pi_phi
        self.pi_u = pi_u

        self.n_layers = n_layers
        self.units = units
        self.learning_rate = learning_rate

    def build(self, hp):
        if self.pi_u or self.pi_phi:
            pi_learning_rate = hp.Float("pi_learning_rate", 1e-6, 1e-3, sampling='log')
            pi_clipnorm = hp.Float("pi_clipnorm", 1e-2, 1e5, sampling='log')

        else:
            # self.n_layers = hp.Int("num_hidden_layers", min_value=1,
            #                 max_value=5, default=1)
            self.n_layers = 5
            self.units = []
            for i in range(self.n_layers):
                self.units.append(hp.Int("units_" + str(i),
                                    min_value=128, max_value=2048, step=1, default=32))
            self.learning_rate = hp.Float("learning_rate", 1e-5, 1e-2, sampling='log')
            pi_learning_rate = None
            pi_clipnorm = None
            

        mlp = MLP(self.pfenv, self.n_layers, self.units, pi_phi=self.pi_phi, pi_u=self.pi_u,
                  pi_learning_rate=pi_learning_rate, pi_clipnorm=pi_clipnorm, alpha=1)
        mlp.compile(self.learning_rate)

        return mlp


class MLPTuner(kt.tuners.BayesianOptimization):
    def run_trial(self, trial, *fit_args, **fit_kwargs):
        # hp = trial.hyperparameters
        batch_size = 2048
        # hp.Int("batch_size", min_value=32,
        #                     max_value=2048, step=32, default=32)
        fit_kwargs['batch_size'] = batch_size
        fit_kwargs['callbacks'] = [
            tf.keras.callbacks.EarlyStopping('val_ME_y', patience=10)]

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
