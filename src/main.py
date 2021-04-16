import potential_flow as pf
from potential_flow import PotentialFlowEnv, SensorArray, sampling
from mlp import MLP
from qm import QM
import numpy as np

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

    samples_u, samples_y = pfenv.sample_sensor_data()
    # sampling.plot(samples_y, "m")

    mlp = MLP(pfenv)
    mlp.compile()
    mlp.fit(samples_u, samples_y, batch_size=32, validation_split=0.2, epochs=10000)
    a = np.array([samples_u[0]])
    # print(mlp.predict(a))
    # print(samples_y[0])
    # qm = QM(pfenv)
    # print(qm.predict(a))


if __name__ == "__main__":
    main()