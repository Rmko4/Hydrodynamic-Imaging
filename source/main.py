import kerastuner
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle

import sampling
from metrics import plot_bar, plot_box_whiskers, plot_prediction_contours, plot_snr
from mlp import MLP, MLPHyperModel, MLPTuner
from polyfit import reduce_polyfit, search_best_model
from potential_flow import PotentialFlowEnv, SensorArray
from qm import QM
from sampling import print_sample_metrics
from utils.mpl_import import *
from utils.mpl_import import cm, plt

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

A_V = 10e-3
A_T = 20e-3

W_T = 0.5

F_V = 45
AMP_V = 2e-3
DUR_V = 1
F_S_V = 2048

FWD_T = 4
BWD_T = 20
MTA_T = np.pi/100

SAMPLE_DISTS = [0.015, 0.03, 0.05]


def gen_poisson_data(pfenv: PotentialFlowEnv, sample_dist, noise=0):
    samples_u, samples_y = pfenv.sample_poisson(
        min_distance=sample_dist, noise_stddev=noise)

    print(sample_dist)
    n_samples = len(samples_u)
    print(n_samples)
    if n_samples < 50000:
        print_sample_metrics(samples_y, [(0, 2), (2, 3)])

    file_name = "sample_pair" + "_" + str(sample_dist) + "_" + str(noise)
    np.savez(DATA_PATH + file_name, samples_u, samples_y)


def gen_path_data(pfenv: PotentialFlowEnv, duration, noise=0):
    samples_u, samples_y = pfenv.sample_path(
        duration=duration, noise_stddev=noise)

    print(duration)
    n_samples = len(samples_u)
    print(n_samples)
    if n_samples < 50000:
        print_sample_metrics(samples_y, [(0, 2), (2, 3)])

    sampling.plot(samples_y)

    file_name = "full_path_" + str(duration) + "_" + str(noise)
    np.savez(DATA_PATH + file_name, samples_u, samples_y)


def gen_poisson_path_data(pfenv: PotentialFlowEnv, sensors: SensorArray, sample_dist, f_s=2048, n_fwd, n_bwd, max_turn_angle, noise=0):
    file_z = "sample_pair_" + str(sample_dist) + "_0"
    _, samples_y = init_data(
        file_z, pfenv, sensors, noise=0, shuffle=False)

    samples_u, samples_y = pfenv.resample_states_to_path(samples_y, sensors, noise, max_turn_angle=max_turn_angle,
                                                         n_fwd=n_fwd, n_bwd=n_bwd, f_s)

    file_name = "path_" + \
        str(sample_dist) + "_" + str(noise)
    np.savez(DATA_PATH + file_name, samples_u, samples_y)


def gen_sinusoid_data(pfenv: PotentialFlowEnv, sensors: SensorArray,
                      sample_dist, A=0.002, f=45, f_s=2048, duration=1, noise=1.5e-5):
    file_z = "sample_pair_" + str(sample_dist) + "_0"
    _, samples_y = init_data(
        file_z, pfenv, sensors, noise=0, shuffle=False)

    samples_u, samples_y = pfenv.resample_states_to_sinusoid(samples_y, sensors, noise_stddev=noise,
                                                             sampling_freq=f_s,
                                                             A=A, f=f, duration=duration)

    file_name = "sinusoid_" + \
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

    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps)


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

        file_name = data_type + '2_' + \
            str(SAMPLE_DISTS[0]) + '_1.5e-05' + FNAME_POSTFIX
        samples_u, samples_y = load_data(DATA_PATH + file_name)

    p_eval, phi_eval = mlp.evaluate_full(samples_u, samples_y)

    file_name = "MLP_" + "test_" + str(SAMPLE_DISTS[0]) + "_" + data_type

    plot_prediction_contours(pfenv, samples_y, p_eval, phi_eval,
                             save_path=PLOT_PATH + file_name + FNAME_FIG_POSTFIX)

    np.savez(RES_PATH + file_name + FNAME_RES_POSTFIX,
             p_eval, phi_eval, samples_y)


def plot_data(pfenv: PotentialFlowEnv):
    files = ["QM_" + "" + str(SAMPLE_DISTS[0]) + "_" + 'sinusoid',
             "MLP_" + "" + str(SAMPLE_DISTS[0]) + "_" + 'sinusoid',
             "MLP_" + "u_" + str(SAMPLE_DISTS[0]) + "_" + 'sinusoid',
             "MLP_" + "phi_" + str(SAMPLE_DISTS[0]) + "_" + 'sinusoid',
             "QM_" + "" + str(SAMPLE_DISTS[0]) + "_" + 'path',
             "MLP_" + "" + str(SAMPLE_DISTS[0]) + "_" + 'path',
             "MLP_" + "u_" + str(SAMPLE_DISTS[0]) + "_" + 'path',
             "MLP_" + "phi_" + str(SAMPLE_DISTS[0]) + "_" + 'path']

    p_evals = []
    phi_evals = []
    p_counts = []
    phi_counts = []

    levels = [0., 0.01, 0.02, 0.04, 0.07, 0.1]

    for file_name in files:
        data = np.load(RES_PATH + file_name + FNAME_RES_POSTFIX)
        p_eval = data['arr_0']
        phi_eval = data['arr_1']
        samples_y = data['arr_2']

        p_count = []
        phi_count = []

        for i in range(1, len(levels)):
            mask = (p_eval < levels[i]) & (p_eval >= levels[i-1])
            p_count.append(100 * np.count_nonzero(mask)/len(p_eval))
            mask = (phi_eval/np.pi < levels[i]
                    ) & (phi_eval/np.pi >= levels[i-1])
            phi_count.append(100 * np.count_nonzero(mask)/len(phi_eval))

        y_eval = (p_eval + phi_eval)/2
        print(file_name)
        print(np.mean(y_eval))
        print(np.std(y_eval))
        print(np.mean(p_eval))
        print(np.mean(phi_eval))
        print()

        p_evals.append(p_eval)
        phi_evals.append(phi_eval)
        p_counts.append(p_count)
        phi_counts.append(phi_count)
        plot_prediction_contours(pfenv, samples_y, p_eval, phi_eval,
                                 save_path=PLOT_PATH + file_name + FNAME_FIG_POSTFIX)
    cntr = plot_prediction_contours(
        pfenv, samples_y, p_eval, phi_eval, save_path=None)

    p_counts = np.array(p_counts).T
    phi_counts = np.array(phi_counts).T

    cmap = cm.get_cmap('viridis')
    plot_bar(levels, p_counts, phi_counts, cmap, cntr)
    plot_box_whiskers()


def plot_snr(pfenv_v: PotentialFlowEnv, pfenv_t: PotentialFlowEnv, sensors: SensorArray):
    signal = init_data('sinusoid' +
                       str(SAMPLE_DISTS[0]) + '_0', pfenv_v, sensors, noise=0, shuffle=False)
    noisy_signal = init_data('sinusoid' +
                             str(SAMPLE_DISTS[0]) + '_1.5e-05', pfenv_v, sensors, noise=0, shuffle=False)
    file_name = 'sinusoid_' + str(SAMPLE_DISTS[0]) + '_snr'
    plot_snr(pfenv_v, 4, signal[1], signal[0], noisy_signal[0],
             save_path=PLOT_PATH + file_name + FNAME_FIG_POSTFIX)

    signal = init_data('sample_pair_' +
                       str(SAMPLE_DISTS[0]) + '_0', pfenv_t, sensors, noise=0, resample=True, shuffle=False)
    noisy_signal = init_data('path_' +
                             str(SAMPLE_DISTS[0]) + '_1.5e-05', pfenv_t, sensors, noise=0, shuffle=False)
    file_name = 'path_' + str(SAMPLE_DISTS[0]) + '_snr'
    plot_snr(pfenv_t, 4, signal[1], signal[0], noisy_signal[0],
             save_path=PLOT_PATH + file_name + FNAME_FIG_POSTFIX)


def main():
    dimensions = (2 * D, D)
    W_v = 2 * np.pi * F_V * AMP_V  # Speed use for the vibration.

    sensors = SensorArray(N_SENSORS, (-0.4*D, 0.4*D))
    pfenv_v = PotentialFlowEnv(dimensions, Y_OFFSET, sensors, A_V, W_v)
    pfenv_t = PotentialFlowEnv(dimensions, Y_OFFSET, sensors, A_T, W_T)

    gen_poisson_data(pfenv_v, SAMPLE_DISTS[0], noise=0)
    gen_sinusoid_data(
        pfenv_v, sensors, SAMPLE_DISTS[0], A=AMP_V, f=F_V, duration=DUR_V, f_s=F_S_V, noise=1.5e-5)
    gen_poisson_path_data(
        pfenv_t, sensors, SAMPLE_DISTS[0], f_s=F_S_V, max_turn_angle=MTA_T, n_bwd=BWD_T, n_fwd=FWD_T, noise=1.5e-5)

    file_z = 'sinusoid' + str(SAMPLE_DISTS[0]) + '_1.5e-05'
    data_type = 'sinusoid'
    file_z = 'path_' + str(SAMPLE_DISTS[0]) + '_1.5e-05'
    data_type = 'path'
    data = init_data(file_z, pfenv_v, sensors, noise=0, shuffle=True)
    find_best_model(pfenv_v, data, max_trials=100,
                          max_epochs=200, validation_split=0.2)
    run_MLP(pfenv_v, data, data_type=data_type)

    file_z = 'path_' + \
        '2_' + str(SAMPLE_DISTS[0]) + '_1.5e-05'
    data = init_data(file_z, pfenv_v, sensors, noise=0, shuffle=True)

    run_QM(pfenv_v, data, data_type='path', multi_process=True)

    file_z = 'sample_pair_' + str(SAMPLE_DISTS[0]) + '_0'
    data = init_data(file_z, pfenv_v, sensors, noise=0, shuffle=True)

    pfenv_v.W = tf.constant(0.5)
    data_path = pfenv_v.resample_points_to_path(
        data[1], sensors, sampling_freq=F_S_V, noise_stddev=1.5e-5, n_fwd=4, n_bwd=20)
    samples_u = reduce_polyfit(data_path[0], -5)
    data = (samples_u, data[1])
    run_MLP(pfenv_v, data, data_type='path')

    file_z = 'sample_pair_' + '2_' + str(SAMPLE_DISTS[0]) + '_0'
    data = init_data(file_z, pfenv_v, sensors, noise=0, shuffle=True)

    pfenv_v.W = tf.constant(0.5)
    file_name = "path_" + str(SAMPLE_DISTS[0]) + "_" + str(1.5e-5) + '_20_L'
    data = init_data(file_name, pfenv_v, sensors, noise=0, shuffle=True)
    run_QM(pfenv_v, data, data_type='path')


if __name__ == "__main__":
    main()
