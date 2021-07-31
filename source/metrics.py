from potential_flow import PotentialFlowEnv
import tensorflow as tf
import numpy as np
from scipy.stats import binned_statistic_2d
from utils.mpl_import import plt


def gather_p(x):
    return tf.gather(x, indices=[0, 1], axis=-1)


def gather_phi(x):
    return tf.gather(x, indices=2, axis=-1)


def ME_p(y_true, y_pred, ord='euclidean'):
    return tf.reduce_mean(E_p(y_true, y_pred, ord=ord))


def ME_phi(y_true, y_pred):
    return tf.reduce_mean(E_phi(y_true, y_pred))


def ME_y(pfenv: PotentialFlowEnv):
    def ME_y(y_true, y_pred):
        return ME_p(y_true, y_pred)/(2 * max(pfenv.dimensions)) + ME_phi(y_true, y_pred)/(2 * np.pi)
    return ME_y


def MSE(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def E_p(y_true, y_pred, ord='euclidean'):
    return tf.norm(gather_p(y_true) - gather_p(y_pred), ord=ord, axis=-1)


def E_phi(y_true, y_pred):
    phi_e = gather_phi(y_true) - gather_phi(y_pred)
    return tf.abs(tf.atan2(tf.sin(phi_e), tf.cos(phi_e)))


def E_phi_2(y_true, y_pred):
    phi_e = gather_phi(y_true) - gather_phi(y_pred)
    phi_e = tf.math.mod(tf.abs(phi_e), 2 * np.pi)
    return phi_e if phi_e < np.pi else 2 * np.pi - phi_e


def binned_stat(pfenv: PotentialFlowEnv, pos, values, statistic="median", cell_size=0.02):
    x = pos[:, 0]
    y = pos[:, 1]

    dmn = pfenv.domains
    dmn[1][0] = dmn[1][0] - cell_size/1.05

    def bin_domain(dmn, cell_size):
        nbins = int((dmn[1] - dmn[0]) / cell_size)
        edges = np.linspace(dmn[0], dmn[1], nbins)
        return edges

    x_edges = bin_domain(dmn[0], cell_size)
    y_edges = bin_domain(dmn[1], cell_size)

    ret = binned_statistic_2d(x, y, values, statistic, bins=[x_edges, y_edges])

    x_cross = (ret.x_edge + 0.5 * cell_size)[:-1]
    y_cross = (ret.y_edge + 0.5 * cell_size)[:-1]
    xv, yv = np.meshgrid(x_cross, y_cross)
    xv = xv.transpose()
    yv = yv.transpose()
    zv = ret.statistic

    return xv, yv, zv


def plot_prediction_contours(pfenv: PotentialFlowEnv, y_bar, p_eval, phi_eval, save_path=None,
                             title=None):
    data = [p_eval, phi_eval/np.pi]
    levels = [0., 0.01, 0.02, 0.04, 0.07, 0.1]
    # levels = [0., 0.01, 0.03, 0.05, 0.09]
    titles = [r"$\mathrm{E}_\mathbf{p}(\mathrm{m})$",
              r"$\mathrm{E}_\varphi(\mathrm{\pi\:rad})$"]
    suptitle = title
    cell_size = 0.02

    return plot_contours(pfenv, y_bar, data, cell_size, levels=levels,
                         suptitle=suptitle, titles=titles, save_path=save_path)


def plot_snr_contours(pfenv: PotentialFlowEnv, y_bar, x_snr, y_snr, save_path=None):
    data = [x_snr, y_snr]
    levels = [0, 10, 30, 50, 70, 90]
    titles = [r"$v_x(\mathrm{dB})$", r"$v_y(\mathrm{dB})$"]
    cell_size = 0.02

    plot_contours(pfenv, y_bar, data, cell_size, levels=levels,
                  titles=titles, save_path=save_path)


def plot_contours(pfenv: PotentialFlowEnv, y_bar, data, cell_size, levels, suptitle=None, titles=None, save_path=None):
    mesh_med = binned_stat(pfenv, y_bar, data, "median", cell_size=cell_size)
    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, sharey=True, figsize=(3.1, 3.5))

    for i in range(2):
        cntr = axes[i].contour(mesh_med[0], mesh_med[1], mesh_med[2][i], linewidths=0.5,
                               colors='k', levels=levels)
        cntr2 = axes[i].contourf(
            mesh_med[0], mesh_med[1], mesh_med[2][i], levels=levels)

        axes[i].set_title(titles[i])
        axes[i].set_ylabel(r"$y(\mathrm{m})$")

        axes[i].set_aspect("equal")

        s_bar = pfenv.sensor()
        axes[i].scatter(s_bar, np.zeros((len(s_bar), )), s=16)

    plt.xticks([-0.5, -0.2, 0., 0.2, 0.5],
               ['-0.5', '-0.2', '0', '0.2', '0.5'])
    plt.yticks([0., 0.2, 0.4], ['0', '0.2', '0.4'])
    axes[1].set_yticks([0.1, 0.3, 0.5], minor=True)

    axes[1].set_xlim((-.5, .5))
    axes[1].set_ylim(bottom=0.)

    axes[1].set_xlabel(r"$x(\mathrm{m})$")

    fig.colorbar(cntr2, ax=axes.ravel().tolist(), location='right',
                 spacing='proportional', fraction=0.1, shrink=0.86)

    fig.suptitle(suptitle)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return cntr2


def plot_snr(pfenv: PotentialFlowEnv, sensor_i, y_bar, signal, noisy_signal, save_path=None):
    n_sensors = len(pfenv.sensor())
    sensor_i_2 = n_sensors + sensor_i

    def snr(sig, noise_sig, index):
        sig = sig[:, index]
        noise = np.abs(sig - noise_sig[:, index])
        sig = np.abs(sig)
        snr = 20 * np.log10(sig / noise)
        return snr

    x_snr = snr(signal, noisy_signal, sensor_i)
    y_snr = snr(signal, noisy_signal, sensor_i_2)

    plot_snr_contours(pfenv, y_bar, x_snr, y_snr, save_path=save_path)
