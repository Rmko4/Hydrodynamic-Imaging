# Use the full module location. Unfortunately it is aliased.

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import sampling


class SensorArray:
    def __init__(self, n_sensors=1, range=(-100, 100), s_bar=None):
        if s_bar is None:
            self.s_bar = SensorArray.uniform_interval_sensors(n_sensors, range)
        else:
            self.s_bar = s_bar

    def uniform_interval_sensors(n_sensors, range):
        if n_sensors == 1:
            s_bar = np.zeros(1)
        else:
            s_bar = np.linspace(range[0], range[1], n_sensors)

        return s_bar

    pass


class PotentialFlowEnv:
    def __init__(self, dimensions=(1000, 500), y_offset=0, sensor: SensorArray = None):
        self.dimensions = dimensions
        self.y_offset = y_offset
        self.sensor = sensor
        self.domains = None

    @tf.function
    def v(self, s, y_bar, a=10., W=100.):
        b = y_bar[0]
        d = y_bar[1]

        phi = y_bar[2]
        rho = (s - b) / d

        c = 0.5 * W * tf.pow(a, 3.) / tf.pow(d, 3.)

        rho_sq = tf.square(rho)
        denum = tf.pow(1 + rho_sq, 2.)

        Psi_e = (2 * rho_sq - 1) / denum
        Psi_o = (-3 * rho) / denum
        Psi_n = (2 - rho_sq) / denum

        cos_phi = tf.cos(phi)
        sin_phi = tf.sin(phi)

        v_x = c * (Psi_e * cos_phi + Psi_o * sin_phi)
        v_y = c * (Psi_o * cos_phi + Psi_n * sin_phi)

        return v_x, v_y

    @tf.function
    def Psi_e(self, rho):
        rho_sq = tf.square(rho)
        return (2 * rho_sq - 1) / tf.pow(1 + rho_sq, 2.5)

    @tf.function
    def Psi_o(self, rho):
        rho_sq = tf.square(rho)
        return (-3 * rho) / tf.pow(1 + rho_sq, 2.5)

    @tf.function
    def Psi_n(self, rho):
        rho_sq = tf.square(rho)
        return (2 - rho_sq) / tf.pow(1 + rho_sq, 2.5)

    @tf.function
    def rho(self, s, b, d):
        return (s - b) / d

    def v_set(self, s_bar, samples):
        s_len = s_bar.shape[0]
        u_bar_s = np.empty([samples.shape[0], 2*s_len])
        for i in range(samples):
            y_bar = samples[i]
            for j in range(s_bar):
                s = s_bar[j]
                # u_bar = tf.TensorArray(tf.float32, size=s_len)
                # print("Tracing with", s, y_bar)
                mu_x, mu_y = self.v(s, y_bar)
                u_bar_s[i][2*j] = mu_x
                u_bar_s[i][2*j + 1] = mu_y

    def sample_sensor_data(self, min_distance=100, k=30):
        if not self.domains:
            x = [-self.dimensions[0]/2, self.dimensions[0]/2]
            y = [self.y_offset, self.y_offset + self.dimensions[1]]
            phi = [0, max(self.dimensions)]
            self.domains = np.array([x, y, phi])

        samples = sampling.poisson_disk_sample(self.domains, min_distance, k)
        samples[:, 2] = 2 * np.pi * samples[:, 2] / (self.domains[2, 1])

        self.v_set(self.sensor.s_bar, samples)


def main():
    pfenv = PotentialFlowEnv()
    vel = pfenv.v(tf.constant(0.), (tf.constant(1.),
                                    tf.constant(1.), tf.constant(0.)))
    print(vel)
    pass


if __name__ == "__main__":
    main()
