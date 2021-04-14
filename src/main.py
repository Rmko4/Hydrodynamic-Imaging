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

    def _cube_root(x):
        sign = tf.sign(x)
        x = sign * tf.pow(abs(x), 1./3.)
        return x

    def _normalize(x, scale):
        c = x.numpy()
        x = _cube_root(x)
        a = x.numpy()
        output = scale * x
        b = output.numpy()
        return output

    def call(inputs):
        u_x, u_y = tf.split(inputs, 2, axis=1)
        scale = np.math.pow(pfenv.y_offset, 3) / (pfenv.W * np.math.pow(pfenv.a, 3))

        u_x = _normalize(u_x, 2*scale)
        u_y = _normalize(u_y, scale)

        outputs = tf.concat([u_x, u_y], 1)
        return outputs

    print(samples_u[0])
    print(call(tf.convert_to_tensor(samples_u))[0])

    mlp = MLP(pfenv)
    mlp.compile()
    mlp.fit(samples_u, samples_y, batch_size=32, validation_split=0.2, epochs=100000)
    a = np.array([samples_u[0]])
    # print(mlp.predict(a))
    # print(samples_y[0])
    # qm = QM(pfenv)
    # print(qm.predict(a))


if __name__ == "__main__":
    main()