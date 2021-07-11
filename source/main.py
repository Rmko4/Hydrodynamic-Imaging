from datetime import datetime
from utils.mpl_import import *
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import kerastuner
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle
from tensorflow import keras

import potential_flow as pf
import sampling
from mlp import MLP, MLPHyperModel, MLPTuner
from polyfit import reduce_polyfit, search_best_model
from potential_flow import (PotentialFlowEnv, SensorArray,
                            plot_prediction_contours)
from qm import QM
from sampling import print_sample_metrics
from utils.mpl_import import plt, cm

DATA_PATH = "sample_data/"
RES_PATH = "results/"
PLOT_PATH = 'plots/'
FNAME_PREFIX = "sample_pair_"
FNAME_POSTFIX = ".npz"
FNAME_RES_POSTFIX = "_res.npz"
FNAME_FIG_POSTFIX = '.pdf'

D = .5
Y_OFFSET = .025
N_SENSORS = 8
SAMPLE_DISTS = [0.015, 0.03, 0.05]


def gen_poisson_data_sets(pfenv: PotentialFlowEnv, sample_dist, noise=0):
    samples_u, samples_y = pfenv.sample_poisson(
        min_distance=sample_dist, noise_stddev=noise)

    print(sample_dist)
    n_samples = len(samples_u)
    print(n_samples)
    if n_samples < 50000:
        print_sample_metrics(samples_y, [(0, 2), (2, 3)])

    file_name = "sample_pair" + "_" + str(sample_dist) + "_" + str(noise)
    np.savez(DATA_PATH + file_name, samples_u, samples_y)


def gen_path_data_sets(pfenv: PotentialFlowEnv, duration, noise=0):
    samples_u, samples_y = pfenv.sample_path(
        duration=duration, noise_stddev=noise)

    print(duration)
    n_samples = len(samples_u)
    print(n_samples)
    if n_samples < 50000:
        print_sample_metrics(samples_y, [(0, 2), (2, 3)])

    sampling.plot(samples_y)

    file_name = "sample_pair_path_" + str(duration) + "_" + str(noise)
    np.savez(DATA_PATH + file_name, samples_u, samples_y)


def gen_sinusoid_data_sets(pfenv: PotentialFlowEnv, sensors: SensorArray,
                           sample_dist, A=0.002, f=45, f_s=2048, duration=1, noise=1.5e-5):
    file_z = "sample_pair_" + str(sample_dist) + "_0"
    _, samples_y = init_data(
        file_z, pfenv, sensors, noise=0, shuffle=False)

    samples_u, samples_y = pfenv.resample_points_to_sinusoid(samples_y, noise_stddev=noise,
                                                             sampling_freq=f_s,
                                                             A=A, f=f, duration=duration)

    file_name = "sample_pair_sinusoid_0.4w_" + \
        str(sample_dist) + "_" + str(noise)
    np.savez(DATA_PATH + file_name, samples_u, samples_y)


def load_data(file):
    data = np.load(file)
    samples_u = data['arr_0']
    samples_y = data['arr_1']
    data.close()
    return samples_u, samples_y


def init_data(file_z, pfenv: PotentialFlowEnv, sensors: SensorArray, noise=0, resample=False, shuffle=False, plot=False):
    file_name = file_z + FNAME_POSTFIX
    samples_u, samples_y = load_data(DATA_PATH + file_name)

    if plot:
        sampling.plot(samples_y)

    if noise != 0 or resample is True:
        samples_u = pfenv.resample_sensor(
            samples_y, sensors, noise_stddev=noise)

    if shuffle:
        samples_u, samples_y = sk_shuffle(samples_u, samples_y)
    return samples_u, samples_y


def find_best_model(pfenv, data, max_trials=10, max_epochs=500, validation_split=0.2):
    u, y = data
    hypermodel = MLPHyperModel(pfenv, pi_u=False, pi_phi=False, n_layers=5, units=[
                               1065, 1930, 968, 923, 1992], learning_rate=7.3e-4)
    tuner = MLPTuner(hypermodel, objective=kerastuner.Objective(
        "val_ME_y",  'min'), max_trials=max_trials, directory='tuner_logs', project_name='MLP_0.015_1e-5_path')
    tuner.search_space_summary()

    tuner.search(u, y, epochs=max_epochs, validation_split=validation_split)
    print()
    # tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps)
    model = tuner.get_best_models()[0]
    # model.save('models/hp_optimized')
    return model


def run_QM(pfenv: PotentialFlowEnv, data, data_type='sinusoid', multi_process=True):
    samples_u, samples_y = data
    qm = QM(pfenv)
    # qm.search_best_model(samples_u, samples_y)
    p_eval, phi_eval = qm.evaluate(
        samples_u, samples_y, multi_process=multi_process)

    file_name = "QM_" + str(SAMPLE_DISTS[0]) + "_" + data_type

    plot_prediction_contours(pfenv, samples_y, p_eval, phi_eval,
                             save_path=PLOT_PATH + file_name + FNAME_FIG_POSTFIX)

    np.savez(RES_PATH + file_name + FNAME_RES_POSTFIX,
             p_eval, phi_eval, samples_y)


def run_MLP(pfenv: PotentialFlowEnv, data, window_size=1, data_type='sinusoid'):
    # mlp = MLP(pfenv, 3, units=[512, 160, 32], physics_informed_u=False,
    #           physics_informed_phi=False, phi_gradient=True, window_size=window_size,
    #           print_summary=True)
    # [2048, 1978, 128, 128, 128]
    mlp = MLP(pfenv, 5, units=[2048, 167, 928, 2048, 678], pi_u=True,
              pi_phi=False, phi_gradient=True, pi_learning_rate=2.9e-4, pi_clipnorm=7.4e01,
              window_size=window_size, print_summary=True)
    mlp.compile(learning_rate=3.1e-3)

    if window_size != 1:
        train, val, test = data
        mlp.fit(train, epochs=1000, validation_data=val,
                callbacks=[tf.keras.callbacks.EarlyStopping('val_ME_y', patience=10)])

        samples_y = tf.constant(np.concatenate(
            [y for _, y in test], axis=0), dtype=tf.float32)
        p_eval, phi_eval = mlp.evaluate_full(test, samples_y)

    else:
        samples_u, samples_y = data
        mlp.fit(samples_u, samples_y, batch_size=2048, validation_split=0.2, epochs=200,
                callbacks=[tf.keras.callbacks.EarlyStopping('val_ME_y', patience=10)])

        file_name = 'sample_pair_sinusoid_0.4w_' + '2_' + \
            str(SAMPLE_DISTS[0]) + '_1.5e-05' + FNAME_POSTFIX
        samples_u, samples_y = load_data(DATA_PATH + file_name)

    p_eval, phi_eval = mlp.evaluate_full(samples_u, samples_y)

    file_name = "MLP_" + "test_" + str(SAMPLE_DISTS[0]) + "_" + data_type

    plot_prediction_contours(pfenv, samples_y, p_eval, phi_eval,
                             save_path=PLOT_PATH + file_name + FNAME_FIG_POSTFIX)

    np.savez(RES_PATH + file_name + FNAME_RES_POSTFIX,
             p_eval, phi_eval, samples_y)


def main():
    D_sensors = D
    dimensions = (2 * D, D)
    y_offset = Y_OFFSET
    a_v = 20e-3  # CHANGE TO 20 for path
    f_v = 45
    Amp_v = 2e-3
    W_v = 2 * np.pi * f_v * Amp_v  # Speed use for the vibration.
    dur_v = 1
    f_s_v = 2048
    W_p = 0.5

    sensors = SensorArray(N_SENSORS, (-0.4*D_sensors, 0.4*D_sensors))
    pfenv = PotentialFlowEnv(dimensions, y_offset, sensors, a_v, W_v)
    # pfenv.show_env()

    # gen_sinusoid_data_sets(pfenv, sensors, SAMPLE_DISTS[1], A=Amp_v, f=f_v, duration=dur_v, f_s=f_s_v, noise=1.5e-5)

    # res = pfenv(tf.constant([[-0.3, 0.4, np.pi/4]]))
    # Change noise to 1.5e-5 and a_v W_v to 0.01

    # signal = init_data('sample_pair_sinusoid_0.4w_' +
    #                          str(SAMPLE_DISTS[0]) + '_0', pfenv, sensors, noise=0, shuffle=False)
    # noisy_signal = init_data('sample_pair_sinusoid_0.4w_' +
    #                          str(SAMPLE_DISTS[0]) + '_1.5e-05', pfenv, sensors, noise=0, shuffle=False)
    # file_name = 'sinusoid_' + str(SAMPLE_DISTS[0]) + '_snr'
    # pf.plot_snr(pfenv, 4, signal[1], signal[0], noisy_signal[0], save_path=PLOT_PATH + file_name + FNAME_FIG_POSTFIX)

    # pfenv.W = tf.constant(W_p)
    # signal = init_data('sample_pair_' +
    #                     str(SAMPLE_DISTS[0]) + '_0', pfenv, sensors, noise=0, resample=True, shuffle=False)
    # noisy_signal = init_data('path_' +
    #  str(SAMPLE_DISTS[0]) + '_1.5e-05', pfenv, sensors, noise=0, shuffle=False)
    # noisy_signal = init_data('sample_pair_' +
    #                          str(SAMPLE_DISTS[0]) + '_0', pfenv, sensors, noise=1.5e-5, shuffle=False)
    # file_name = 'path_' + str(SAMPLE_DISTS[0]) + '_snr'
    # pf.plot_snr(pfenv, 4, signal[1], signal[0], noisy_signal[0],
    #             save_path=PLOT_PATH + file_name + FNAME_FIG_POSTFIX)

    # samples_y = signal[1]
    # path_u, path_y = pfenv.resample_points_to_path(samples_y, sensors, noise_stddev=0, n_fwd=4, n_bwd=20)
    # file_z = DATA_PATH + 'path_20_L_2_0' + FNAME_POSTFIX
    # np.savez(file_z, path_u)

    # file_z = DATA_PATH + 'path_20_L_0' + FNAME_POSTFIX
    # data = np.load(file_z)
    # path_u = data['arr_0']
    # data.close()

    # path_u = pfenv.apply_gauss_noise(path_u, 1.5e-5)
    # u_pred = reduce_polyfit(path_u, -5, 7.09e-09, 5.58e-11, 1.29e-08)
    # file_z = DATA_PATH + 'path_' + str(SAMPLE_DISTS[0]) + '_1.5e-05_20_L_2' + FNAME_POSTFIX
    # np.savez(file_z, u_pred, samples_y)

    # path_u_noise = pfenv.apply_gauss_noise(path_u, 1.5e-5)
    # path_u_noise, u_true = sk_shuffle(path_u_noise, signal[0])

    # search_best_model(path_u_noise[0:5000], u_true[0:5000])

    # search = HalvingRandomSearchCV(PolyFit(), param_distributions, cv=2, scoring='neg_mean_squared_error', n_candidates=3).fit(path_u_noise, path_u)
    # print(search.best_params_)

    # np.savez(DATA_PATH + 'path_' + str(SAMPLE_DISTS[0]) + '_1.5e-05_NEW', samples_u, samples_y)
    pass

    # file_z = 'sample_pair_sinusoid_0.4w_' + \
    #     str(SAMPLE_DISTS[0]) + '_0'
    # data = init_data(file_z, pfenv, sensors, noise=0, shuffle=False)
    # file_z = 'sample_pair_sinusoid_0.4w_' + \
    #     str(SAMPLE_DISTS[0]) + '_1.5e-05'
    # data2 = init_data(file_z, pfenv, sensors, noise=1.5e-5, shuffle=False)
    # pass

    # file_name = "QM_" + "" + \
    #     str(SAMPLE_DISTS[0]) + "_path"

    # files = ["QM_" + "" + str(SAMPLE_DISTS[0]) + "_" + 'sinusoid',
    #          "MLP_" + "" + str(SAMPLE_DISTS[0]) + "_" + 'sinusoid',
    #          "MLP_" + "u_" + str(SAMPLE_DISTS[0]) + "_" + 'sinusoid',
    #          "MLP_" + "phi_" + str(SAMPLE_DISTS[0]) + "_" + 'sinusoid',
    #          "QM_" + "" + str(SAMPLE_DISTS[0]) + "_" + 'path',
    #          "MLP_" + "" + str(SAMPLE_DISTS[0]) + "_" + 'path',
    #          "MLP_" + "u_" + str(SAMPLE_DISTS[0]) + "_" + 'path',
    #          "MLP_" + "phi_" + str(SAMPLE_DISTS[0]) + "_" + 'path']

    # p_evals = []
    # phi_evals = []
    # p_counts = []
    # phi_counts = []

    # levels = [0., 0.01, 0.02, 0.04, 0.07, 0.1]

    # for file_name in files:
    #     data = np.load(RES_PATH + file_name + FNAME_RES_POSTFIX)
    #     p_eval = data['arr_0']
    #     phi_eval = data['arr_1']
    #     samples_y = data['arr_2']

    #     p_count = []
    #     phi_count = []

    #     for i in range(1, len(levels)):
    #         mask = (p_eval < levels[i]) & (p_eval >= levels[i-1])
    #         p_count.append(100 * np.count_nonzero(mask)/len(p_eval))
    #         mask = (phi_eval/np.pi < levels[i]
    #                 ) & (phi_eval/np.pi >= levels[i-1])
    #         phi_count.append(100 * np.count_nonzero(mask)/len(phi_eval))

    #     # print(np.mean(p_eval))
    #     # print(np.mean(phi_eval))

    #     p_evals.append(p_eval)
    #     phi_evals.append(phi_eval)
    #     p_counts.append(p_count)
    #     phi_counts.append(phi_count)
    #     plot_prediction_contours(pfenv, samples_y, p_eval, phi_eval,
    #                             save_path=PLOT_PATH + file_name + FNAME_FIG_POSTFIX)
    # cntr = plot_prediction_contours(pfenv, samples_y, p_eval, phi_eval, save_path=None)

    # p_counts = np.array(p_counts).T
    # phi_counts = np.array(phi_counts).T

    # cmap = cm.get_cmap('viridis')
    # # plt.figure(figsize=(6.69, C_WIDTH))
    # fig, axes = plt.subplots(
    #     nrows=1, ncols=2, sharex=True, sharey=True, figsize=(C_WIDTH, C_WIDTH + 2))
    # axes[0].grid(axis='y', lw=1)
    # axes[1].grid(axis='y', lw=1)

    # for i in range(len(levels) - 1):
    #     bottom = 0 if i-1 < 0 else bottom + p_counts[i-1]
    #     axes[0].bar([0,1,2,3,5,6,7,8], p_counts[i], bottom=bottom, color=cmap(
    #         (levels[i] + 0.5*(levels[i+1]-levels[i]))/levels[-1]), edgecolor='white',
    #         linewidth=0.5)
    # for i in range(len(levels) - 1):
    #     bottom = 0 if i-1 < 0 else bottom + phi_counts[i-1]
    #     axes[1].bar([0,1,2,3,5,6,7,8], phi_counts[i], bottom=bottom, color=cmap(
    #         (levels[i] + 0.5*(levels[i+1]-levels[i]))/levels[-1]), edgecolor='white',
    #         linewidth=0.5)

    # plt.xticks([0, 1, 2, 3, 5, 6, 7, 8],
    #            2 * ['QM', 'MLP', r'MLP-$\mathbf{u}$', r'MLP-$\mathbf{\varphi}$'], fontsize=9)
    # titles = [r"$\mathrm{E}_\mathbf{p}$", r"$\mathrm{E}_\varphi/\pi$"]

    # y_pos = [0.93, 0.85]

    # for i in range(2):
    #     axes[i].set_axisbelow(True)
    #     plt.setp(axes[i].get_xticklabels(), rotation=90, horizontalalignment='center', va='top')
    #     axes[i].set_title(titles[i])
    #     axes[i].annotate('Vib', xy=(0.25, y_pos[i]), xytext=(0.25, y_pos[i] + 0.05), xycoords='axes fraction',
    #         ha='center', va='bottom',
    #         bbox=dict(boxstyle='square', fc='white', ec='white'),
    #         arrowprops=dict(arrowstyle='-[, widthB=1.85, lengthB=0.5', lw=1.0))
    #     axes[i].annotate('Tra', xy=(0.76, y_pos[i]), xytext=(0.76, y_pos[i] + 0.05), xycoords='axes fraction',
    #         ha='center', va='bottom',
    #         bbox=dict(boxstyle='square', fc='white', ec='white'),
    #         arrowprops=dict(arrowstyle='-[, widthB=1.85, lengthB=0.5', lw=1.0))

    # axes[0].set_ylim((0., 100))
    # axes[0].set_ylabel(r"Rel. frequency (%)")
    # axes[0].yaxis.set_label_coords(-.21, 0.50)

    # cb = fig.colorbar(cntr, ax=axes.ravel().tolist(), location='top',
    #              spacing='proportional', shrink=1.0, pad=0.1)
    # plt.setp(cb.ax.get_xticklabels(),rotation=90, ha='center', va='bottom')
    # # plt.tight_layout(rect=(-0.02,0,1,1))
    # plt.savefig(PLOT_PATH + "spread" + FNAME_FIG_POSTFIX, bbox_inches='tight')
    # plt.show()

    # fig, axes = plt.subplots(
    #     nrows=1, ncols=2, sharex=True, sharey=True, figsize=(C_WIDTH, C_WIDTH + 1))
    # axes[0].grid(axis='y', which='both', lw=1)
    # axes[1].grid(axis='y', which='both', lw=1)

    # titles = [r"$\mathrm{E}_\mathbf{p}$", r"$\mathrm{E}_\varphi/\pi$"]

    # y_pos = [0.66, 0.56, 0.97, 0.845]
    # data = [np.array(p_evals).T, np.array(phi_evals).T / np.pi]

    # for i in range(2):
    #     for j in range(len(levels) - 1):
    #         axes[i].bar(4, levels[j + 1] - levels[j], bottom=levels[j], color=cmap(
    #             (levels[j] + 0.5*(levels[j+1]-levels[j]))/levels[-1]), width=9)

    #     axes[i].hlines([0, 0.02, 0.04, 0.06, 0.08, 0.1], -0.5, 8.5, colors='darkgrey', linewidths=1)
    #     axes[i].hlines([0.01, 0.07], -0.5, 8.5, colors='darkgrey', linewidths=0.5)
    #     axes[i].boxplot(data[i], showfliers=False, positions=[
    #                     0, 1, 2, 3, 5, 6, 7, 8], patch_artist=True, boxprops=dict(facecolor='C0'))
    #     axes[i].set_axisbelow(True)



    #     axes[i].set_title(titles[i])
    #     if i == 0:
    #         axes[i].annotate('Vib', xy=(0.25, y_pos[2*i]), xytext=(0.25, y_pos[2*i] + 0.05), xycoords='axes fraction',
    #                          ha='center', va='bottom',
    #                          bbox=dict(boxstyle='square', fc='white', ec='white'),
    #                          arrowprops=dict(arrowstyle='-[, widthB=1.85, lengthB=0.5', lw=1.0))
    #     else:
    #         axes[i].annotate('', xy=(0.25, y_pos[2*i]), xytext=(0.25, y_pos[2*i] + 0.035), xycoords='axes fraction',
    #                          ha='center', va='bottom',
    #                          arrowprops=dict(arrowstyle='-[, widthB=1.85, lengthB=0.5', lw=1.0))
    #         axes[i].annotate('Vib', xy=(0.25, y_pos[2*i]), xytext=(0.15, y_pos[2*i] + 0.04), xycoords='axes fraction',
    #                          ha='center', va='bottom',
    #                          bbox=dict(boxstyle='square', fc='white', ec='white'))
    #     axes[i].annotate('Tra', xy=(0.76, y_pos[2*i + 1]), xytext=(0.76, y_pos[2*i + 1] + 0.05), xycoords='axes fraction',
    #                      ha='center', va='bottom',
    #                      bbox=dict(boxstyle='square', fc='white', ec='white'),
    #                      arrowprops=dict(arrowstyle='-[, widthB=1.85, lengthB=0.5', lw=1.0))

    # axes[0].set_ylabel(r"(m)")
    # axes[1].set_ylabel(r"(rad)")
    # axes[0].set_ylim((-0.005, 0.210))

    # plt.xticks([0, 1, 2, 3, 5, 6, 7, 8],
    #            2 * ['QM', 'MLP', r'MLP-$\mathbf{u}$', r'MLP-$\mathbf{\varphi}$'], fontsize=9)
    # axes[i].yaxis.set_major_locator(MultipleLocator(0.04))
    # axes[i].yaxis.set_minor_locator(MultipleLocator(0.02))

    # for i in range(2):
    #     plt.setp(axes[i].get_xticklabels(), rotation=90,
    #              horizontalalignment='center', va='top')

    # plt.savefig(PLOT_PATH + "boxplot" + FNAME_FIG_POSTFIX, bbox_inches='tight')
    # plt.show()

    file_z = 'sample_pair_sinusoid_0.4w_' + str(SAMPLE_DISTS[0]) + '_1.5e-05'
    data_type = 'sinusoid'
    # file_z = 'path_' + str(SAMPLE_DISTS[0]) + '_1.5e-05'
    # data_type = 'path'
    data = init_data(file_z, pfenv, sensors, noise=0, shuffle=True)
    # mlp = find_best_model(pfenv, data, max_trials=100, max_epochs=200, validation_split=0.2)
    run_MLP(pfenv, data, data_type=data_type)

    # file_z = 'path_' + \
    #     '2_' + str(SAMPLE_DISTS[0]) + '_1.5e-05'
    # data = init_data(file_z, pfenv, sensors, noise=0, shuffle=True)

    # run_QM(pfenv, data, data_type='path', multi_process=True)

    # file_z = 'sample_pair_' + str(SAMPLE_DISTS[0]) + '_0'
    # data = init_data(file_z, pfenv, sensors, noise=0, shuffle=True)

    # pfenv.W = tf.constant(0.5)
    # data_path = pfenv.resample_points_to_path(data[1], sensors, sampling_freq=f_s_v, noise_stddev=1.5e-5, n_fwd=4, n_bwd=20)
    # samples_u = reduce_polyfit(data_path[0], -5)
    # data = (samples_u, data[1])

    # run_MLP(pfenv, data, data_type='path')

    # file_z = 'sample_pair_' + '2_' + str(SAMPLE_DISTS[0]) + '_0'
    # data = init_data(file_z, pfenv, sensors, noise=0, shuffle=True)

    # pfenv.W = tf.constant(0.5)
    # # data_path = pfenv.resample_points_to_path(data[1], sensors, sampling_freq=f_s_v, noise_stddev=1.0e-5, n_fwd=4, n_bwd=20)
    # # samples_u = reduce_polyfit(data_path[0], -5)
    # # data = (samples_u, data[1])
    # file_name = "path_" + str(SAMPLE_DISTS[0]) + "_" + str(1.5e-5) + '_20_L'
    # # np.savez(DATA_PATH + file_name, data[0], data[1])
    # data = init_data(file_name, pfenv, sensors, noise=0, shuffle=True)

    # run_QM(pfenv, data, data_type='path')

    # file_z = 'sample_pair_' + str(SAMPLE_DISTS[2]) + '_0' # ADD BACK _2
    # u, y = init_data(file_z, pfenv, sensors, noise=0, shuffle=True)

    # pfenv.W = tf.constant(0.5)
    # data = pfenv.resample_points_to_path(y, sensors, sampling_freq=f_s_v, noise_stddev=1.5e-5, n_fwd=4, n_bwd=20)

    # qm = QM(pfenv)
    # qm.search_best_model(data[0], y)

    # pass


if __name__ == "__main__":
    main()


def out():
    pass
    # data_train = init_data("path_2500.0", pfenv, sensors, 1e-5)
    # data_test = init_data("path_2_2500.0", pfenv, sensors, 1e-5)

    # from tensorflow import keras

    # def window_generator(data, window_length=3, sub_indices=slice(None), shuffle=True):
    #     samples_u, samples_y = data

    #     samples_u = samples_u[sub_indices]
    #     samples_y = samples_y[sub_indices]

    #     samples_y_r = np.roll(samples_y, window_length - 1, axis=0)
    #     ds = keras.preprocessing.timeseries_dataset_from_array(
    #         samples_u, samples_y_r, sequence_length=window_length,
    #         batch_size=128, shuffle=shuffle)
    #     return ds

    # def split_window_generator(data, window_length=3, validation_split=0.2):
    #     n = len(data[0])
    #     cut_off_i = int((1 - validation_split) * n)
    #     train = window_generator(data, window_length, slice(0, cut_off_i))
    #     val = window_generator(data, window_length, slice(
    #         cut_off_i, None), shuffle=False)
    #     return train, val

    # data_gen_triple = (*split_window_generator(data_train, 5), window_generator(data_test, 5, shuffle=False))
    # run_MLP(pfenv, sensors, data_gen_triple, 5)

    # gen_path_data_sets(pfenv, 10.0, 0)
    # gen_poisson_data_sets(pfenv, SAMPLE_DISTS[0], 0)

    # import matplotlib.pyplot as plt
    # import scipy.signal as signal
    # err = 0

    # for k in range(len(samples_u)):
    #     u_bar_s = samples_u[k]
    #     # plt.plot(u_bar_s[0], label='low')
    #     # plt.plot(u_bar_s[-1], label='high')
    #     # plt.plot(pfenv.v_np(pfenv.sensor(), *samples_y[0]), label="true")
    #     # mavg = np.mean(u_bar_s, axis=0)
    #     # plt.plot(mavg, label="mean")
    #     # B, A = signal.butter(3, 0.1)
    #     # low_pass = signal.filtfilt(B,A, u_bar_s, axis=0, padlen=3)
    #     # plt.plot(low_pass[-5], label="low_pass")
    #     u_real = pfenv(tf.constant(y_path[k], dtype=tf.float32)).numpy()
    #     for i in range(3):
    #         plt.plot(u_bar_s[:, i], label='measured signal')
    #         plt.plot(u_real[:, i], label="true signal")
    #         t = np.linspace(0, 24, 25)
    #         # for j in range(4):
    #         #     res = np.polyfit(t, u_bar_s[:, i], j, full=True)
    #         #     p = np.poly1d(res[0])
    #         #     plt.plot(p(t), label=str(j))

    #         residual = 1
    #         residuals = np.empty(4)
    #         for j in [0, 1, 2, 3]:
    #             res = np.polyfit(t, u_bar_s[:, i], j, full=True)
    #             delta_residual = residual - res[1][0]
    #             if j == 0 or (residual > 2.67e-9 and delta_residual > 3.25e-10 ):
    #                 p = np.poly1d(res[0])
    #             residual = res[1][0]
    #             residuals[j] = res[1][0]
    #         if residual > 5.75e-9:
    #             plt.plot(u_bar_s[:, i], label=str("best fit - max"))
    #         else:
    #             plt.plot(p(t), label=str("best fit"))

    #         plt.legend()
    #         plt.show()
    # print(err)

    # sampling.plot(np.reshape(y_path, (-1, 3))[0:20])
