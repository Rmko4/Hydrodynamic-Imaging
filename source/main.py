import kerastuner
from polyfit import reduce_polyfit
import potential_flow as pf
from potential_flow import PotentialFlowEnv, SensorArray, plot_prediction_contours
from mlp import MLP, MLPHyperModel, MLPTuner
from qm import QM
import sampling
import numpy as np
from datetime import datetime
import tensorflow as tf
from utils.mpl_import import plt
from tensorflow import keras
from sklearn.utils import shuffle as sk_shuffle
from sampling import print_sample_metrics

DATA_PATH = "sample_data/"
RES_PATH = "results/"
FNAME_PREFIX = "sample_pair_"
FNAME_PREFIX_SIN = "sample_pair_sinusoid_0.4w_"
FNAME_POSTFIX = ".npz"
FNAME_POSTFIX_SIN = ".npz"
FNAME_RES_POSTFIX = "_res.npz"

D = .5
Y_OFFSET = .025
N_SENSORS = 8
SAMPLE_DISTS = [0.015, 0.015]


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


def init_data(file_z, pfenv: PotentialFlowEnv, sensors: SensorArray, noise=1e-5, shuffle=False, plot=False):
    file_name = file_z + FNAME_POSTFIX
    samples_u, samples_y = load_data(DATA_PATH + file_name)

    if plot:
        sampling.plot(samples_y)

    if noise != 0:
        samples_u = pfenv.resample_sensor(
            samples_y, sensors, noise_stddev=noise)

    if shuffle:
        samples_u, samples_y = sk_shuffle(samples_u, samples_y)
    return samples_u, samples_y


def find_best_model(pfenv, data, max_trials=10, max_epochs=500, validation_split=0.2):
    u, y = data
    hypermodel = MLPHyperModel(pfenv, False, True, 1)
    tuner = MLPTuner(hypermodel, objective=kerastuner.Objective(
        "val_ME_y",  'min'), max_trials=max_trials, directory='tuner_logs', project_name='MLP_0.015_1e-5_full_range')
    tuner.search_space_summary()

    tuner.search(u, y, epochs=max_epochs, validation_split=validation_split)
    print()
    # tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps)
    model = tuner.get_best_models()[0]
    # model.save('models/hp_optimized')
    return model


def run_QM(pfenv: PotentialFlowEnv, data):
    samples_u, samples_y = data
    qm = QM(pfenv)
    # qm.search_best_model(samples_u, samples_y)
    p_eval, phi_eval = qm.evaluate(samples_u, samples_y)
    plot_prediction_contours(pfenv, samples_y, p_eval, phi_eval)


def run_MLP(pfenv: PotentialFlowEnv, sensors: SensorArray, data, window_size=1):
    # 3, units=[512, 160, 32]
    mlp = MLP(pfenv, 5, units=[544, 1372, 1150, 2048, 1725], physics_informed_u=False,
              physics_informed_phi=False, phi_gradient=True, window_size=window_size,
              print_summary=True)
    mlp.compile()
    # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
    #                                                  histogram_freq=1)
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
        samples_u2, samples_y2 = load_data(DATA_PATH + file_name)
        p_eval, phi_eval = mlp.evaluate_full(samples_u2, samples_y2)
        plot_prediction_contours(pfenv, samples_y2, p_eval, phi_eval)
        mlp.physics_informed_u = True
        mlp.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-5))
        mlp.fit(samples_u, samples_y, batch_size=2048, validation_split=0.2, epochs=200,
                callbacks=[tf.keras.callbacks.EarlyStopping('val_ME_y', patience=10)])
        file_name = 'sample_pair_sinusoid_0.4w_' + '2_' + \
            str(SAMPLE_DISTS[0]) + '_1.5e-05' + FNAME_POSTFIX
        samples_u, samples_y = load_data(DATA_PATH + file_name)
        # samples_u = pfenv.resample_sensor(
        #     samples_y, sensors, noise_stddev=1e-5)

        p_eval, phi_eval = mlp.evaluate_full(samples_u, samples_y)

    # file_name = FNAME_PREFIX + "2_" + str(SAMPLE_DISTS[1]) + FNAME_RES_POSTFIX
    # np.savez(RES_PATH + file_name, p_eval, phi_eval, samples_y)

    plot_prediction_contours(pfenv, samples_y, p_eval, phi_eval)


def main():
    D_sensors = D
    dimensions = (2 * D, D)
    y_offset_v = Y_OFFSET
    a_v = 10e-3
    f_v = 45
    Amp_v = 2e-3
    W_v = 2 * np.pi * f_v * Amp_v
    dur_v = 1
    f_s_v = 2048

    sensors = SensorArray(N_SENSORS, (-0.4*D_sensors, 0.4*D_sensors))
    pfenv = PotentialFlowEnv(dimensions, y_offset_v, sensors, a_v, W_v)
    # pfenv.show_env()

    # res = pfenv(tf.constant([[-0.3, 0.4, np.pi/4]]))
    # Change noise to 1.5e-5 and a_v W_v to 0.01

    # gen_sinusoid_data_sets(pfenv, sensors, SAMPLE_DISTS[0], A=Amp_v, f=f_v, duration=dur_v, f_s=f_s_v, noise=1.5e-5)
    # signal = init_data('sample_pair_sinusoid_0.4w_' +
    #      str(SAMPLE_DISTS[0]) + '_0', pfenv, sensors, noise=0, shuffle=False)

    # noisy_signal = init_data('sample_pair_sinusoid_0.4w_' +
    #                          str(SAMPLE_DISTS[0]) + '_1.5e-05', pfenv, sensors, noise=0, shuffle=False)

    # pf.plot_snr(pfenv, 4, signal[1], signal[0], noisy_signal[0])
    # path_u, path_y = pfenv.resample_points_to_path(samples_y, sensors, noise_stddev=1e-5, n_fwd=4, n_bwd=20)
    # samples_u = reduce_polyfit(path_u, -5)
    # data = (samples_u, samples_y)

    # run_MLP(pfenv, sensors, noisy_signal)

    # mlp = find_best_model(pfenv, data, max_trials=100, max_epochs=200, validation_split=0.2)
    # pass

    # run_QM(pfenv, noisy_signal)


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
