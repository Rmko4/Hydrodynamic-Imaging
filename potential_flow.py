# Use the full module location. Unfortunately it is aliased.

#import tensorflow.compat.v1 as tf
import tensorflow_core._api.v1.compat.v1 as tf
#import tensorflow.keras
import tensorflow_core.python.keras.api._v1.keras as keras
import numpy as np
import matplotlib.pyplot as plt

class PotentialFlowEnv:
    def __init__(self, dimensions=(500, 1000), y_offset=0, sensor=None):
        self.dimensions = dimensions
        self.y_offset = y_offset
        self.sensor = sensor

    def sample_sensor_data():
        pass

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

# function to get sensor at equaly spaced intervals within range

# function to get training samples
# Size of the sampling position range
# Number of samples or blue noise sampling distance.


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

def main():
    tf.enable_eager_execution()
    das = PotentialFlowEnv.v(tf.constant(
        0.), (tf.constant(1.), tf.constant(1.), tf.constant(0.)))
    print(das[0])
    pass


if __name__ == "__main__":
    main()