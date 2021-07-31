# Use the full module location. Unfortunately it is aliased.

import numpy as np
import tensorflow as tf
from matplotlib import patches

import sampling
from utils.mpl_import import C_WIDTH, plt


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


class PotentialFlowEnv:
    Y_BAR_SIZE = 3

    def __init__(self, dimensions=(1, .5), y_offset=0, sensor: SensorArray = None, a=.025, W=.25):
        self.dimensions = dimensions
        self.y_offset = y_offset

        x = [-self.dimensions[0]/2, self.dimensions[0]/2]
        y = [self.y_offset, self.y_offset + self.dimensions[1]]
        phi = [0, max(self.dimensions)]
        self.domains = np.array([x, y, phi])

        self.initSensor(sensor)

        self.a = tf.constant(a)
        self.W = tf.constant(W)

        self.C_dw = 0.5 * tf.pow(a, 3.)
        self.C_d = W * self.C_dw
        self.C_d_np = self.C_d.numpy()

    def __call__(self, y: tf.Tensor):
        return tf.vectorized_map(self.v_tf, y)

    @ tf.function
    def v_tf(self, y_bar):
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

    def v_tf_g_vectorized(self, g: tf.Tensor):
        return tf.vectorized_map(self.v_tf_g, g)

    def v_tf_g(self, g_bar):
        s = self.sensor()

        b = g_bar[0]
        d = g_bar[1]
        phi = g_bar[2]
        W = g_bar[3]

        rho = (s - b) / d

        c = W * self.C_dw / tf.pow(d, 3.)

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

    def v_np(self, s, b, d, phi):
        rho = (s - b) / d

        c = self.C_d_np / d**3

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

    def initSensor(self, sensor):
        if sensor is None:
            self.sensor = SensorArray()
        else:
            self.sensor = sensor

    def sample_poisson(self, sensor: SensorArray = None, noise_stddev=0, min_distance=.1, k=30):
        if sensor is not None:
            self.sensor = sensor
        # Write to tensor array
        samples_y = sampling.poisson_disc_sample(
            self.domains, min_distance, k, wrap_domains=[2])
        samples_y[:, 2] = 2 * np.pi * \
            samples_y[:, 2] / (self.domains[2, 1])

        samples_u = self.resample_sensor(samples_y, self.sensor, noise_stddev)
        return samples_u, samples_y

    def sample_path(self, sensor: SensorArray = None, noise_stddev=0,
                    sampling_freq=2048.0, inner_sampling_factor=10, duration=20.0,
                    max_turn_angle=np.pi/100, circum_radius=None, mode="rotate"):
        if sensor is not None:
            self.sensor = sensor
        n_samples = int(sampling_freq * duration)
        step_distance = self.W.numpy() / (sampling_freq * inner_sampling_factor)
        samples_y = sampling.sample_path_2D(
            self.domains, step_distance, max_turn_angle=max_turn_angle,
            circum_radius=circum_radius, inner_sampling_factor=inner_sampling_factor,
            n_samples=n_samples, mode=mode)

        samples_u = self.resample_sensor(samples_y, self.sensor, noise_stddev)
        return samples_u, samples_y

    def resample_sensor(self, samples_y, sensor: SensorArray = None, noise_stddev=0):
        if sensor is not None:
            self.sensor = sensor

        flat_y = samples_y.reshape((-1, 3))
        samples_u = self(tf.constant(flat_y, tf.float32)).numpy()
        samples_u = self.apply_gauss_noise(samples_u, noise_stddev)

        new_shape = (*samples_y.shape[:-1], samples_u.shape[-1])
        samples_u = samples_u.reshape(new_shape)

        return samples_u

    def apply_gauss_noise(self, samples_u, noise_stddev=1.5e-5):
        if noise_stddev != 0:
            gaus_noise = sampling.rng.normal(
                0, noise_stddev, samples_u.shape).astype(np.float32)
            return samples_u + gaus_noise
        return samples_u

    def resample_states_to_path(self, samples_y, sensor: SensorArray = None,
                                noise_stddev=0, sampling_freq=2048.0,
                                inner_sampling_factor=10, n_fwd=4, n_bwd=20,
                                max_turn_angle=np.pi/100):
        if sensor is not None:
            self.sensor = sensor
        step_distance = self.W.numpy() / (sampling_freq * inner_sampling_factor)
        samples_y = sampling.sample_path_on_pos(samples_y, step_distance=step_distance,
                                                max_turn_angle=max_turn_angle,
                                                n_fwd=n_fwd, n_bwd=n_bwd,
                                                inner_sampling_factor=inner_sampling_factor)

        samples_u = self.resample_sensor(samples_y, self.sensor, noise_stddev)

        return samples_u, samples_y

    def resample_states_to_sinusoid(self, samples_y, sensor: SensorArray = None,
                                    noise_stddev=0, sampling_freq=2048, A=0.002,
                                    f=45, duration=1, batch_size=4096):
        if sensor is not None:
            self.sensor = sensor

        window_len = duration * sampling_freq
        t = np.linspace(0, duration, int(sampling_freq), endpoint=False)

        phase_p = np.sin(2 * np.pi * f * t)
        W_n = 2 * np.pi * f * A * np.cos(2 * np.pi * f * t)

        hamm = np.hamming(window_len).reshape(-1, 1)
        A_correction = 2 / np.sum(hamm)

        def p(b, d, phi):
            b = b + A * np.cos(phi) * phase_p
            d = d + A * np.sin(phi) * phase_p
            return b, d

        ds = tf.data.Dataset.from_tensor_slices(samples_y)
        ds = ds.batch(batch_size)

        samples_u = []

        for batch_y in ds:
            y_bar_s = []
            for y_bar in batch_y:
                b = y_bar[0]
                d = y_bar[1]
                phi = y_bar[2]

                b_n, d_n = p(b, d, phi)
                phi_n = np.repeat(phi, window_len)
                g_bar = np.stack((b_n, d_n, phi_n, W_n), axis=-1)

                g_bar = tf.constant(g_bar, dtype=tf.float32)
                y_bar_s.append(g_bar)

            y_bar_s = np.array(y_bar_s).reshape(-1, 4)
            u_bar_s = self.v_tf_g_vectorized(y_bar_s).numpy()
            u_bar_s = self.apply_gauss_noise(u_bar_s, noise_stddev)

            u_bar_s = u_bar_s.reshape((-1, window_len, u_bar_s.shape[-1]))
            hamm_u = hamm * u_bar_s

            x = A_correction * np.fft.rfft(hamm_u, axis=1)[:, f, :]

            magn = np.abs(x)
            phase = np.angle(x)

            samples_u.append(np.sign(0.5 * np.pi - np.abs(phase)) * magn)

        samples_u = np.concatenate(samples_u)

        return samples_u, samples_y

    # Gives best results with the default pfenv params
    def show_env(self, figsize=(C_WIDTH, 2.2)):
        plt.figure(figsize=figsize)
        ax = plt.gca()

        if self.domains is not None:
            plt.hlines(self.domains[1, :], self.domains[0, 0],
                       self.domains[0, 1], colors='grey', linestyles='--', zorder=-2)
            plt.vlines(self.domains[0, :], self.domains[1, 0],
                       self.domains[1, 1], colors='grey', linestyles='--', zorder=-2)

        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_bounds((0, 1.02*self.domains[1, 1]))

        s_bar = self.sensor()
        plt.scatter(s_bar, np.zeros((len(s_bar), )), s=16, zorder=-2)

        p = (0.1, 0.25)
        plt.scatter(*p, edgecolors='black')

        s_x = s_bar[4]
        s_y = 0

        plt.arrow(*p, s_x-p[0], s_y-p[1], length_includes_head=True,
                  head_width=0.015, head_length=0.03, fc='black', zorder=-1)
        plt.arrow(*p, 0.1, 0.1, length_includes_head=True,
                  head_width=0.015, head_length=0.03, fc='black', zorder=-1)

        plt.vlines(p[0], 0, p[1], linestyles='dashed',
                   colors='black', linewidth=1, zorder=-1)
        plt.hlines(p[1], 0, p[0] + 0.1, linestyles='dashed',
                   colors='black', linewidth=1, zorder=-1)

        plt.xticks([-0.5, -0.2, 0., p[0], 0.2, 0.5],
                   ['-500', '-200', '$O$', '$b$', '200', '500'])
        plt.yticks([0.2, p[1], 0.4], ['200', '$d$', '400'])

        arc = patches.Arc(p, 0.15, 0.15, 0, 0, 45)
        ax.add_patch(arc)

        angle = (np.math.atan2(p[1] - s_y, p[0] - s_x) + np.pi) * 180/np.pi
        arc = patches.Arc(p, 0.1, 0.1, 0, angle, 45)
        ax.add_patch(arc)

        plt.annotate(r'$\mathbf{w}$', (0.11, 0.305))
        plt.annotate(r'$\mathbf{r}$', (0.03, 0.13))
        plt.annotate(r'$\varphi$', (0.18, 0.28))
        plt.annotate(r'$\theta$', (0.14, 0.18))

        plt.ylim(0, 1.15*self.domains[1, 1])
        ax.set_aspect("equal")
        ax.spines['top'].set_visible(False)

        plt.annotate(text='', xy=(0.96*self.domains[0, 0], self.domains[1, 0]), xytext=(0.96*self.domains[0, 0],
                                                                                        self.domains[1, 1]), arrowprops=dict(arrowstyle='<->'))
        plt.annotate(text='', xy=(self.domains[0, 0], 1.04*self.domains[1, 1]), xytext=(
            self.domains[0, 1], 1.04*self.domains[1, 1]), arrowprops=dict(arrowstyle='<->'))
        plt.annotate('500 mm', (0.88*self.domains[0, 0], (self.domains[1, 1] -
                                                         self.domains[1, 0])/2), rotation=90, ha='center', va='center')
        plt.annotate(
            '1000 mm', (0, 1.1*self.domains[1, 1]), ha='center', va='center')

        plt.xlabel(r'$x(\mathrm{mm}) \longrightarrow$')
        plt.ylabel(r'$y(\mathrm{mm}) \longrightarrow$')

        plt.tight_layout()
        plt.show()


def main():
    tf.config.run_functions_eagerly(True)

    print(ME_phi(tf.constant([0., 0., -1.5*np.pi]),
                 tf.constant([0., 0., 2.2*np.pi])))

    print(E_phi_2(tf.constant([0., 0., -2.2*np.pi]),
                  tf.constant([0., 0., 2.2*np.pi])))

    pfenv = PotentialFlowEnv(sensor=SensorArray(10, (-0.5, 0.5)))
    y_bar = tf.constant([.5, .5, 0.5*np.pi])

    g_bar = tf.concat((y_bar, [1.]), axis=-1)
    print(pfenv.v_tf_g(g_bar))

    print(pfenv.v(tf.constant(0.), (tf.constant(.5),
                                    tf.constant(.5), tf.constant(0.))))
    print(pfenv.v(tf.constant(-.25), (tf.constant(.5),
                                      tf.constant(.5), tf.constant(0.))))



if __name__ == "__main__":
    main()
