import potential_flow as pf
from potential_flow import PotentialFlowEnv, SensorArray

D = 500
Y_OFFSET = 25
N_SENSORS = 8

if __name__ == "__main__":
    pf.main()
    D_v = D
    y_offset_v = Y_OFFSET
    D_sensors = 0.4 * D
    dimensions = (D_v, 2 * D_v)

    sensors = SensorArray(N_SENSORS, (-D_sensors, D_sensors))
    pfenv = PotentialFlowEnv(dimensions, y_offset_v, sensors)
    
    pass