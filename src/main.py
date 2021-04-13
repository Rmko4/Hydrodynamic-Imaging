import potential_flow as pf
from potential_flow import PotentialFlowEnv, SensorArray, sampling
from mlp import MLP

D = 500
Y_OFFSET = 25
N_SENSORS = 8

if __name__ == "__main__":
    pf.main()
    D_v = D
    y_offset_v = Y_OFFSET
    D_sensors = 0.4 * D
    dimensions = (2 * D_v, D_v)

    sensors = SensorArray(N_SENSORS, (-D_sensors, D_sensors))
    pfenv = PotentialFlowEnv(dimensions, y_offset_v, sensors)

    samples_u, samples_y = pfenv.sample_sensor_data()
    sampling.plot(samples_y, "mm")

    mlp = MLP(pfenv)
    mlp.fit(samples_u, samples_y, batch_size=64, validation_split=0.2, epochs=100)


    
    pass