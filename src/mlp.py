import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from potential_flow import PotentialFlowEnv, SensorArray


class MLP(keras.Sequential):
    def __init__(self, pfenv: PotentialFlowEnv):
        super(MLP, self).__init__()
        self.pfenv = pfenv
        s_bar = pfenv.getSensor().s_bar
        input_shape = (2*s_bar.size,)

        self.add(layers.Dense(200, activation="sigmoid", input_shape=input_shape))
        self.add(layers.Dense(pfenv.Y_BAR_SIZE))
        self.summary()

        self.compile(
            optimizer=keras.optimizers.SGD(),  # Optimizer
            # Loss function to minimize
            loss=keras.losses.MeanSquaredError(),
            # List of metrics to monitor
            metrics=[keras.metrics.MeanSquaredError()]
        )


def main():
    m = MLP(PotentialFlowEnv(sensor=SensorArray(8)))


if __name__ == "__main__":
    main()
