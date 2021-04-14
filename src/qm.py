import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from potential_flow import PotentialFlowEnv, SensorArray

class QM:
    def __init__(self, pfenv : PotentialFlowEnv):
        self.pfenv = pfenv
        s_bar = pfenv.getSensor().s_bar
        input_shape = (2*s_bar.size,)

    def psi_quad_sq(v_x, v_y):
        return v_x * v_x + 0.5 * v_y * v_y

    def _step_predict(self, u_bar):
        return u_bar

    def predict(self, samples_u):
        samples_y = []
        for u_bar in samples_u:
            y_bar = self._step_predict(u_bar)
            samples_y.append(y_bar)

        return np.array(samples_y)