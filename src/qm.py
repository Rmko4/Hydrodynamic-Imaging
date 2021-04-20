from potential_flow import PotentialFlowEnv
import numpy as np


class QM:
    def __init__(self, pfenv: PotentialFlowEnv):
        self.pfenv = pfenv
        self.s_bar = pfenv.sensor().numpy()

    def psi_quad(self, v_x, v_y):
        return np.sqrt(v_x**2 + 0.5 * v_y**2)

    def anch(self, quad, s_bar, b_i, anch_height):
        i = b_i
        sensor_len = s_bar.size
        while i < sensor_len and quad[i] > anch_height:
            i += 1
        if i == sensor_len:
            anch_p = None
        else:
            x_interpolated = (
                anch_height - quad[i - 1]) / (quad[i] - quad[i - 1])
            anch_p = s_bar[i - 1] + x_interpolated * (s_bar[i] - s_bar[i - 1])
        return anch_p

    def phi_calc(self, v_x_bar, b, d):
        phi_estimates = []
        for v_x, s in zip(v_x_bar, self.s_bar):
            rho = self.pfenv.rho(s, b, d)

            Psi_e = self.pfenv.Psi_e(rho)
            Psi_o = self.pfenv.Psi_o(rho)
            Psi_env = np.sqrt(Psi_e**2 + Psi_o**2)

            # Psi_n = self.pfenv.Psi_n(rho)
            phi_prime = np.math.atan(Psi_o/Psi_e)
            alpha = np.math.acos(v_x/Psi_env)

            phi_estimates.append(phi_prime + alpha)
            phi_estimates.append(phi_prime - alpha)

        return phi_estimates[0]

    def _step_predict(self, u_bar):
        u_xy = np.split(u_bar, 2)
        quad = self.psi_quad(u_xy[0], u_xy[1])
        b_i = np.argmax(quad)
        b = self.s_bar[b_i]

        anch_height = 0.458 * quad[b_i]
        anch_plus = self.anch(quad, self.s_bar, b_i, anch_height)
        anch_min = self.anch(np.flip(quad), np.flip(
            self.s_bar), self.s_bar.size - b_i - 1, anch_height)

        if anch_plus is None and anch_min is None:
            d = (self.s_bar[-1] - self.s_bar[0]) / 1.79
        elif anch_plus is None:
            d = 2 * (b - anch_min) / 1.79
        elif anch_min is None:
            d = 2 * (anch_plus - b) / 1.79
        else:
            d = (anch_plus - anch_min) / 1.79

        phi = self.phi_calc(u_xy[0], b, d)

        return [b, d, phi]

    def predict(self, samples_u):
        samples_y = []
        for u_bar in samples_u:
            y_bar = self._step_predict(u_bar)
            samples_y.append(y_bar)

        return np.array(samples_y)



def main():
    from potential_flow import SensorArray
    import tensorflow as tf

    D = .5
    Y_OFFSET = .025
    N_SENSORS = 8

    D_sensors = 0.5 * D
    dimensions = (2 * D, D)
    y_offset_v = Y_OFFSET
    a_v = 0.05 * D
    W_v = 0.5 * D

    sensors = SensorArray(N_SENSORS, (-D_sensors, D_sensors))
    pfenv = PotentialFlowEnv(dimensions, y_offset_v, sensors, a_v, W_v)

    y_bar = tf.constant([.5, .5, 2.])

    result = pfenv.forward_step(y_bar)
    qm = QM(pfenv)
    qm.phi_calc(result[:N_SENSORS].numpy(), .5, .5)

if __name__ == "__main__":
    main()