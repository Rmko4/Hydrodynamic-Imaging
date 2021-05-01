# Use the full module location. Unfortunately it is aliased.

import tensorflow as tf
import numpy as np
import sampling
import matplotlib.pyplot as plt
import poisson_disc


class SensorArray:
    def __init__(self, n_sensors=1, range=(-.1, .1), s_bar=None):
        if s_bar is None:
            self.s_bar = SensorArray.uniform_interval_sensors(n_sensors, range)
        else:
            self.s_bar = s_bar

    def __call__(self) -> tf.Tensor:
        return self.s_bar

    def uniform_interval_sensors(n_sensors, range):
        if n_sensors == 1:
            s_bar = tf.constant([0.])
        else:
            s_bar = tf.linspace(range[0], range[1], n_sensors)

        return s_bar

    pass


class PotentialFlowEnv:
    Y_BAR_SIZE = 3

    def __init__(self, dimensions=(1, .5), y_offset=0, sensor: SensorArray = None, a=.025, W=.25):
        self.dimensions = dimensions
        self.y_offset = y_offset
        x = [-self.dimensions[0]/2, self.dimensions[0]/2]
        y = [self.y_offset, self.y_offset + self.dimensions[1]]
        phi = [0, max(self.dimensions)]
        self.sample_domains = np.array([x, y, phi])
        self.initSensor(sensor)

        self.a = tf.constant(a)
        self.W = tf.constant(W)
        self.C_d = 0.5 * W * tf.pow(a, 3.)

    def __call__(self, y: tf.Tensor):
        return tf.vectorized_map(self.v_tf_vectorized, y)

    @ tf.function
    def v(self, s, y_bar):
        b = y_bar[0]
        d = y_bar[1]
        phi = y_bar[2]

        rho = (s - b) / d

        c = self.C_d / tf.pow(d, 3.)

        rho_sq = tf.square(rho)
        denum = tf.pow(1 + rho_sq, 2.5)

        Psi_e = (2 * rho_sq - 1) / denum
        Psi_o = (-3 * rho) / denum
        Psi_n = (2 - rho_sq) / denum

        cos_phi = tf.cos(phi)
        sin_phi = tf.sin(phi)

        v_x = c * (Psi_e * cos_phi + Psi_o * sin_phi)
        v_y = c * (Psi_o * cos_phi + Psi_n * sin_phi)

        return v_x, v_y

    @ tf.function
    def v_tf_vectorized(self, y_bar):
        s = self.sensor()

        b = y_bar[0]
        d = y_bar[1]
        phi = y_bar[2]

        rho = (s - b) / d

        c = self.C_d / tf.pow(d, 3.)

        rho_sq = tf.square(rho)
        denum = tf.pow(1 + rho_sq, 2.5)

        Psi_e = (2 * rho_sq - 1) / denum
        Psi_o = (-3 * rho) / denum
        Psi_n = (2 - rho_sq) / denum

        cos_phi = tf.cos(phi)
        sin_phi = tf.sin(phi)

        v_x = c * (Psi_e * cos_phi + Psi_o * sin_phi)
        v_y = c * (Psi_o * cos_phi + Psi_n * sin_phi)

        return tf.concat([v_x, v_y], 0)

    def v_np(self, s, b, d, phi, y_bar=None):
        if y_bar is not None:
            b = y_bar[0]
            d = y_bar[1]
            phi = y_bar[2]
        rho = (s - b) / d

        c = self.C_d.numpy() / d**3

        rho_sq = np.square(rho)
        denum = np.power(1 + rho_sq, 2.5)

        Psi_e = (2 * rho_sq - 1) / denum
        Psi_o = (-3 * rho) / denum
        Psi_n = (2 - rho_sq) / denum

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        v_x = c * (Psi_e * cos_phi + Psi_o * sin_phi)
        v_y = c * (Psi_o * cos_phi + Psi_n * sin_phi)

        return np.concatenate((v_x, v_y))

    def Psi_e(self, rho):
        rho_sq = tf.square(rho)
        return (2 * rho_sq - 1) / tf.pow(1 + rho_sq, 2.5)

    def Psi_o(self, rho):
        rho_sq = tf.square(rho)
        return (-3 * rho) / tf.pow(1 + rho_sq, 2.5)

    def Psi_n(self, rho):
        rho_sq = tf.square(rho)
        return (2 - rho_sq) / tf.pow(1 + rho_sq, 2.5)

    def rho(self, s, b, d):
        return (s - b) / d

    def sample_pairs(self, noise_stddev=0, min_distance=.1, k=30):
        # Write to tensor array
        samples_y = sampling.poisson_disc_sample(
            self.sample_domains, min_distance, k)
        samples_y[:, 2] = 2 * np.pi * \
            samples_y[:, 2] / (self.sample_domains[2, 1])
        samples_y = tf.constant(samples_y, tf.float32)

        samples_u = self(samples_y)
        gaus_noise = tf.random.normal(tf.shape(samples_u), 0, noise_stddev)
        samples_u += gaus_noise

        self.samples = (samples_u, samples_y)
        return samples_u.numpy(), samples_y.numpy()

    def sample_sensor_data(self, samples_y, sensor: SensorArray = None, noise_stddev=0):
        if sensor is not None:
            self.sensor = sensor

        samples_u = self(samples_y)
        # sampling.rng.normal(0, noise_stddev, np.shape(samples_u))
        gaus_noise = tf.random.normal(tf.shape(samples_u), 0, noise_stddev)
        samples_u += gaus_noise

        return samples_u.numpy()

    def initSensor(self, sensor):
        if sensor is None:
            self.sensor = SensorArray()
        else:
            self.sensor = sensor


def gather_p(x):
    return tf.gather(x, indices=[0, 1], axis=-1)


def gather_phi(x):
    return tf.gather(x, indices=2, axis=-1)


def ME_p(y_true, y_pred):
    return tf.reduce_mean(E_p(y_true, y_pred))


def ME_phi(y_true, y_pred):
    return tf.reduce_mean(E_phi(y_true, y_pred))


def ME_y(pfenv: PotentialFlowEnv):
    def ME_y(y_true, y_pred):
        return ME_p(y_true, y_pred)/(2 * max(pfenv.dimensions)) + ME_phi(y_true, y_pred)/(2 * np.pi)
    return ME_y


def E_p(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(gather_p(y_true) - gather_p(y_pred)), axis=-1))


def E_phi(y_true, y_pred):
    phi_e = gather_phi(y_true) - gather_phi(y_pred)
    return tf.abs(tf.atan2(tf.sin(phi_e), tf.cos(phi_e)))


def MSE(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def plot_prediction_contours(pfenv: PotentialFlowEnv, y_bar, p_eval, phi_eval):
    data = [p_eval, phi_eval/np.pi]
    titles = [r"$\mathrm{E}_\mathbf{p}$", r"$\mathrm{E}_\phi/\pi$"]
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(9,9))
    
    axes[1].set_xlabel("x")

    for i in range(2):
        axes[i].tricontour(y_bar[:, 0], y_bar[:, 1], data[i], linewidths=0.5,
                    colors='k', levels=[0.0, 0.01, 0.03, 0.05, 0.1])
        cntr = axes[i].tricontourf(y_bar[:, 0], y_bar[:, 1], data[i], levels=[
                            0.0, 0.01, 0.03, 0.05, 0.1])
        axes[i].set_title(titles[i])
        axes[i].set_ylabel("y")
        axes[i].set_aspect("equal")
        # SET equal aspect
        s_bar = pfenv.sensor()
        axes[i].scatter(s_bar, np.zeros((len(s_bar), )))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cntr, cax=cbar_ax)

    fig.suptitle("Multilayer Perceptron - Noise 1e-5")
    plt.show()

def main():
    tf.config.run_functions_eagerly(True)

    print(ME_phi(tf.constant([0., 0., -0.5]), tf.constant([0., 0., 48.])))

    # pfenv = PotentialFlowEnv(sensor=SensorArray(1000, (-0.5, 0.5)))
    # y_bar = tf.constant([.5, .5, 2.])

    # print(y_bar)
    # print(pfenv(tf.constant([[.5, .5, 2.]])))

    # result = pfenv([y_bar])

    # u_xy = np.split(result, 2)
    # t = np.linspace(-0.5, 0.5, 1000)

    # plt.plot(t, u_xy[0])
    # plt.plot(t, u_xy[1])
    # plt.show()

    # print(pfenv.v(tf.constant(0.), (tf.constant(.5),
    #                                 tf.constant(.5), tf.constant(0.))))
    # print(pfenv.v(tf.constant(-.25), (tf.constant(.5),
    #                                   tf.constant(.5), tf.constant(0.))))
    # pass


if __name__ == "__main__":
    main()
