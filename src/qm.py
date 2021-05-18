from potential_flow import ME_phi, ME_p, ME_y, E_p, E_phi, PotentialFlowEnv, plot_prediction_contours
from potential_flow import SensorArray
import numpy as np
import tensorflow as tf
from scipy.stats import circmean
from scipy.optimize import curve_fit
import scipy.signal as signal
# from scipy.optimize._lsq.trf import trf_bounds
# from scipy.optimize._numdiff import approx_derivative


class QM:
    def __init__(self, pfenv: PotentialFlowEnv, optimization_iter=5):
        self.pfenv = pfenv
        self.s_bar = pfenv.sensor().numpy()
        self.optimization_iter = optimization_iter

    def psi_quad(self, u_bar):
        u_xy = np.split(u_bar, 2)
        return np.sqrt(u_xy[0]**2 + 0.5 * u_xy[1]**2)

    def psi_quad_env(self, s, b, d, phi):
        u_bar = self.pfenv.v_np(s, b, d, phi)
        return self.psi_quad(u_bar)

    def psi_quad_scaled_f(self, scale):
        def f(s, b, d, phi):
            u_bar = self.pfenv.v_np(s, b, d, phi)
            return scale * self.psi_quad(u_bar)
        return f

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

    def phi_calc(self, u_xy, b, d):
        u_xy = np.split(u_xy, 2)
        phi_estimates = []
        for v_x, v_y, s in zip(u_xy[0], u_xy[1], self.s_bar):
            rho = self.pfenv.rho(s, b, d)

            Psi_e = self.pfenv.Psi_e(rho).numpy()
            Psi_o = self.pfenv.Psi_o(rho).numpy()
            Psi_n = self.pfenv.Psi_n(rho).numpy()

            cos_phi = v_x * Psi_n - v_y * Psi_o
            sin_phi = v_y * Psi_e - v_x * Psi_o

            phi = np.pi + np.math.atan2(sin_phi, cos_phi)
            phi_estimates.append(phi)

        return circmean(phi_estimates)

    def _step_predict(self, u_bar_s, curve_fit=True, true_idx=-5, alpha=3.0e-9, beta=2e-10):
        if u_bar_s.ndim != 1:
            win_len = np.shape(u_bar_s)[0]
            signal_len = np.shape(u_bar_s)[1]
            # u_bar = signal.decimate(u_bar_s, win_len, axis=-1)
            # B, A = signal.butter(3, 0.2)
            # low_pass = signal.filtfilt(B,A, u_bar_s, axis=0)
            # u_bar = low_pass[-5]
            # u_bar = np.mean(u_bar_s, axis=0)
            u_bar = np.empty(signal_len)
   
            for i in range(signal_len):
                residual = 1
                for j in range(win_len):
                    t = np.linspace(0, win_len - 1, win_len)
                    res = np.polyfit(t, u_bar_s[:, i], j, full=True)
                    if res[1].size != 0:
                        delta_residual = residual - res[1][0]
                        residual = res[1][0]
                        if delta_residual < beta and res[1][0] < alpha:
                            break
                    else:
                        p = np.poly1d(res[0])
                        break
                    p = np.poly1d(res[0])
                u_bar[i] = p(t[true_idx])

                # for j in range(0, 20):
                #     t = np.linspace(0, win_len - 1, win_len)
                #     res = np.polyfit(t, u_bar_s[:, i], j, full=True)
                #     if delta_residual < 2e-10 and res[1][0] < 2.5e-9:
                #         break
                #     p = np.poly1d(res[0])
                # u_bar[i] = p(t[true_idx])
        else:
            u_bar = u_bar_s

        quad = self.psi_quad(u_bar)
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

        d_min = self.pfenv.sample_domains[1, 0]
        d_max = self.pfenv.sample_domains[1, 1]
        if d > d_max:
            d = d_max
        elif d < d_min:
            d = d_min

        phi = self.phi_calc(u_bar, b, d)
        p0 = [b, d, phi]
        if curve_fit:
            y_bar = self.curve_fit_optimize(p0, u_bar, quad)
        else:
            y_bar = p0

        return y_bar

    def curve_fit_optimize(self, p0, u_bar, quad):
        dmn = self.pfenv.sample_domains
        bounds = ([dmn[0, 0], dmn[1, 0], 0], [dmn[0, 1], dmn[1, 1], 2 * np.pi])
        scale = 1 / np.abs(quad).max()
        quad *= scale
        p_i = p0

        # def func_wrapped(params):
        #     return self.psi_quad_scaled_f(scale)(self.s_bar, *params) - quad

        # def jac_wrapped(x, f):
        #     J = approx_derivative(self.psi_quad_scaled_f(scale), x, rel_step=1e-3, method='2-point',
        #                             f0=f, bounds=bounds)
        #     if J.ndim != 2:  # J is guaranteed not sparse.
        #         J = np.atleast_2d(J)

        #     return J

        # J0 = jac_wrapped(p_i, quad)

        # d = trf_bounds(func_wrapped, jac_wrapped, p_i, quad, J0, bounds[0], bounds[1], 1e-08, 1e-03, 1e-08, 10,
        #        np.array([1., 1., 1.]), None, 'exact', {}, 0)
        for _ in range(self.optimization_iter):
            # curve_fit uses version that does not raise error upon reaching iter beyond max_nfev
            p_i, _ = curve_fit(self.psi_quad_scaled_f(scale), self.s_bar, quad, p0=p_i,
                               method='trf', bounds=bounds, xtol=1e-3, diff_step=1e-3, max_nfev=10)
            p_i[2] = self.phi_calc(u_bar, p_i[0], p_i[1])
        return p_i

    def predict(self, samples_u, curve_fit=True, true_idx=-5):
        samples_y = []
        for u_bar in samples_u:
            y_bar = self._step_predict(u_bar, curve_fit, true_idx)
            samples_y.append(y_bar)

        return np.array(samples_y)

    def evaluate(self, samples_u, samples_y, curve_fit=True, true_idx=-5):
        pred_y = self.predict(samples_u, curve_fit, true_idx)
        pred_y = tf.convert_to_tensor(pred_y, dtype=tf.float32)
        p_eval = E_p(samples_y, pred_y) # Select correct pred_y out of complete signal
        phi_eval = E_phi(samples_y, pred_y)
        print(ME_p(samples_y, pred_y))
        print(ME_phi(samples_y, pred_y))
        print(ME_y(self.pfenv)(samples_y, pred_y))
        return p_eval, phi_eval


def main():
    tf.config.run_functions_eagerly(True)
    D = .5
    Y_OFFSET = .025
    N_SENSORS = 8

    D_sensors = D
    dimensions = (2 * D, D)
    y_offset_v = Y_OFFSET
    a_v = 0.05 * D
    W_v = 0.5 * D

    sensors = SensorArray(N_SENSORS, (-D_sensors, D_sensors))
    pfenv = PotentialFlowEnv(dimensions, y_offset_v, sensors, a_v, W_v)

    y_bar = tf.constant([[.5, .5, 2.]])
    result = pfenv(y_bar).numpy()
    qm = QM(pfenv)
    print(qm.predict(result))

    # qm.phi_calc(result, .5, .5)

    samples_u, samples_y = pfenv.sample_poisson(noise_stddev=1e-5)
    # print(qm.psi_quad_env(pfenv.sensor().numpy(), samples_y[0][0].numpy(
    # ), samples_y[0][1].numpy(), samples_y[0][2].numpy()))
    # print(pfenv.forward_step(samples_y[0]))
    p_eval, phi_eval = qm.evaluate(samples_u, samples_y)
    plot_prediction_contours(pfenv, samples_y, p_eval, phi_eval)


if __name__ == "__main__":
    main()
