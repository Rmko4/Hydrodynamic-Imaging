import potential_flow as pf
from potential_flow import PotentialFlowEnv, SensorArray, sampling
from mlp import MLP
from qm import QM
import numpy as np
from datetime import datetime
import tensorflow as tf

from sampling import print_sample_metrics

DATA_PATH = "sample_data/"

D = .5
Y_OFFSET = .025
N_SENSORS = 8
SAMPLE_DISTS = [0.03]


def gen_data_sets(pfenv: PotentialFlowEnv, sample_dists, noise):
    for sample_dist in sample_dists:
        samples_u, samples_y = pfenv.sample_sensor_data(
            min_distance=sample_dist, noise_stddev=noise)
        print(sample_dist)
        print(len(samples_u))
        print_sample_metrics(samples_y, [(0, 2), (2, 3)])
        file_name = "sample_pair_2_" + "_" + str(sample_dist) + "_" + str(noise)
        np.savez(DATA_PATH + file_name, samples_u, samples_y)


def main():
    D_sensors = 0.5 * D
    dimensions = (2 * D, D)
    y_offset_v = Y_OFFSET
    a_v = 0.05 * D
    W_v = 0.5 * D

    sensors = SensorArray(N_SENSORS, (-D_sensors, D_sensors))
    pfenv = PotentialFlowEnv(dimensions, y_offset_v, sensors, a_v, W_v)

    gen_data_sets(pfenv, SAMPLE_DISTS, 0)

    # print(pfenv(samples_y))
    # qm = QM(pfenv)
    # print(qm.evaluate(samples_u, samples_y))

    # sampling.plot(samples_y, "m")

    mlp = MLP(pfenv)
    mlp.compile(physics_informed=False, alpha=1)
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
