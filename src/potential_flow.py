# Use the full module location. Unfortunately it is aliased.

import tensorflow as tf
import numpy as np
import sampling


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
        self.domains = None
        self.initSensor(sensor)

        self.a = tf.constant(a)
        self.W = tf.constant(W)
        self.C_d = 0.5 * W * tf.pow(a, 3.)

    @tf.function
    def __call__(self, y_bar: tf.Tensor):
        s_bar = self.sensor()
        size = tf.size(s_bar)
        ta = tf.TensorArray(y_bar.dtype, 2*size)
        for i in range(size):
            v_x, v_y = self.v(s_bar[i], y_bar)
            ta = ta.write(i, v_x)
            ta = ta.write(size + i, v_y)
        return ta.stack()

    @tf.function
    def v(self, s, y_bar):
        b = y_bar[0]
        d = y_bar[1]
        phi = y_bar[2]

        rho = (s - b) / d

        c = self.C_d / tf.pow(d, 3.)

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

    def v_set(self, s_bar, samples_y):
        samples_u = []
        for i in range(len(samples_y)):
            y_bar = tf.convert_to_tensor(samples_y[i], dtype=tf.float32)
            u_x_bar = []
            u_y_bar = []
            # TODO: Use call
            for j in range(len(s_bar)):
                s = tf.convert_to_tensor(s_bar[j], dtype=tf.float32)
                mu_x, mu_y = self.v(s, y_bar)
                u_x_bar.append(mu_x)
                u_y_bar.append(mu_y)

            samples_u.append(u_x_bar + u_y_bar)
        return samples_u

    def sample_sensor_data(self, min_distance=.1, k=30):
        if not self.domains:
            x = [-self.dimensions[0]/2, self.dimensions[0]/2]
            y = [self.y_offset, self.y_offset + self.dimensions[1]]
            phi = [0, max(self.dimensions)]
            self.domains = np.array([x, y, phi])
        # Write to tensor array
        samples_y = sampling.poisson_disk_sample(self.domains, min_distance, k)
        samples_y[:, 2] = 2 * np.pi * samples_y[:, 2] / (self.domains[2, 1])

        samples_u = self.v_set(self.sensor(), samples_y)
        samples_u = np.array(samples_u)
        self.samples = (samples_u, samples_y)
        return samples_u, samples_y

    def initSensor(self, sensor):
        if sensor is None:
            self.sensor = SensorArray()
        else:
            self.sensor = sensor


def main():
    tf.config.run_functions_eagerly(True)
    pfenv = PotentialFlowEnv(sensor=SensorArray(8))
    print(pfenv(tf.constant([0., 0.5, 1.5])))
    print(pfenv.v(tf.constant(0.), (tf.constant(.05),
                                    tf.constant(.1), tf.constant(0.))))
    print(pfenv.v(tf.constant(-.25), (tf.constant(.5),
                                      tf.constant(.5), tf.constant(0.))))
    pass


if __name__ == "__main__":
    main()
