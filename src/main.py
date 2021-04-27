from kerastuner.tuners.randomsearch import RandomSearch
from kerastuner.tuners.hyperband import Hyperband
import kerastuner
import potential_flow as pf
from potential_flow import PotentialFlowEnv, SensorArray, plot_prediction_contours, sampling
from mlp import MLP, MLPHyperModel, MLPTuner
from qm import QM
import numpy as np
from datetime import datetime
import tensorflow as tf
from sklearn.utils import shuffle as sk_shuffle

from sampling import print_sample_metrics

DATA_PATH = "sample_data/"
FNAME_PREFIX = "sample_pair_"
FNAME_POSTFIX = "_0.npz"

D = .5
Y_OFFSET = .025
N_SENSORS = 16
SAMPLE_DISTS = [0.05]


def gen_data_sets(pfenv: PotentialFlowEnv, sample_dists, noise):
    for sample_dist in sample_dists:
        samples_u, samples_y = pfenv.sample_pairs(
            min_distance=sample_dist, noise_stddev=noise)
        print(sample_dist)
        print(len(samples_u))
        print_sample_metrics(samples_y, [(0, 2), (2, 3)])
        file_name = "sample_pair_2" + "_" + str(sample_dist) + "_" + str(noise)
        np.savez(DATA_PATH + file_name, samples_u, samples_y)


def load_data(file):
    data = np.load(file)
    samples_u = data['arr_0']
    samples_y = data['arr_1']
    data.close()
    return samples_u, samples_y

def find_best_model(pfenv, data):
    u, y = data
    hypermodel = MLPHyperModel(pfenv, False, 1)
    tuner = MLPTuner(hypermodel, objective=kerastuner.Objective(
        "val_ME_y",  'min'), max_trials=50, directory='tuner_logs', project_name='MLP_0.05_1e-5')
    tuner.search_space_summary()

    tuner.search(u, y, epochs=2000, validation_split=0.2)
    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps)
    model = tuner.get_best_models()[0]
    model.save('models/hp_optimized')

def main():
    D_sensors = D
    dimensions = (2 * D, D)
    y_offset_v = Y_OFFSET
    a_v = 0.05 * D
    W_v = 0.5 * D

    sensors = SensorArray(N_SENSORS, (-D_sensors, D_sensors))
    pfenv = PotentialFlowEnv(dimensions, y_offset_v, sensors, a_v, W_v)

    # gen_data_sets(pfenv, SAMPLE_DISTS, 0)

    file_name = FNAME_PREFIX + str(SAMPLE_DISTS[0]) + FNAME_POSTFIX
    samples_u, samples_y = load_data(DATA_PATH + file_name)

    # sampling.plot(samples_y, "m")

    samples_u = pfenv.sample_sensor_data(samples_y, sensors, noise_stddev=1e-5)
    samples_u, samples_y = sk_shuffle(samples_u, samples_y)

    find_best_model(pfenv, (samples_u, samples_y))

    # qm = QM(pfenv)
    # p_eval, phi_eval = qm.evaluate(samples_u, samples_y, True)
    # plot_prediction_contours(pfenv, samples_y, p_eval, phi_eval)


    # mlp = MLP(pfenv)
    # mlp.compile(physics_informed=False, alpha=1)
    # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
    #                                                 histogram_freq = 1, profile_batch="2,50")
    # , callbacks = [tboard_callback]

    # mlp.fit(samples_u, samples_y, batch_size=32, validation_split=0.2, epochs=100000)
    # a = np.array([samples_u[0]])
    # print(mlp.predict(a))
    # print(samples_y[0])


if __name__ == "__main__":
    main()
