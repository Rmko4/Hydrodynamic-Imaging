import potential_flow as pf
from potential_flow import PotentialFlowEnv, SensorArray, sampling
from mlp import MLP
from qm import QM
import numpy as np
from datetime import datetime

import tensorflow as tf

D = .5
Y_OFFSET = .025
N_SENSORS = 8

def main():
    D_sensors = 0.5 * D
    dimensions = (2 * D, D)
    y_offset_v = Y_OFFSET
    a_v = 0.05 * D
    W_v = 0.5 * D

    sensors = SensorArray(N_SENSORS, (-D_sensors, D_sensors))
    pfenv = PotentialFlowEnv(dimensions, y_offset_v, sensors, a_v, W_v)

    samples_u, samples_y = pfenv.sample_sensor_data(noise_stddev=1e-5)
    qm = QM(pfenv)
    qm.predict(samples_u.numpy())
    # sampling.plot(samples_y, "m")

    # mlp = MLP(pfenv)
    # mlp.compile(physics_informed=False)
    # logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
    #                                                 histogram_freq = 1, profile_batch="2,50")
    # , callbacks = [tboard_callback]

    
    # mlp.fit(samples_u, samples_y, batch_size=32, validation_split=0.2, epochs=10000)
    # a = np.array([samples_u[0]])
    # print(mlp.predict(a))
    # print(samples_y[0])
    # qm = QM(pfenv)
    # print(qm.predict(a))


if __name__ == "__main__":
    main()