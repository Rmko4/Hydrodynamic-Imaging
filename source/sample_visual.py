import numpy as np

import sampling
from potential_flow import PotentialFlowEnv, SensorArray
from utils.mpl_import import plt

D = .5
Y_OFFSET = .025
N_SENSORS = 8
A = 10e-3
W_T = 5e-1


def plot_f(plt):
    plt.figure(figsize=(3.229, 2.0))
    plt.xticks([-0.5, -0.2, 0., 0.2, 0.5],
               ['-0.5', '-0.2', '0', '0.2', '0.5'])
    plt.yticks([0., 0.2, 0.4], ['0', '0.2', '0.4'])

    ax = plt.gca()

    plt.xlabel(r'$x(\mathrm{m})$')
    plt.ylabel(r'$y(\mathrm{m})$')

    ax.xaxis.set_label_coords(0.5, -.17)
    ax.yaxis.set_label_coords(-.11, 0.55)


def plot_toy_poisson(pfenv: PotentialFlowEnv):
    _, samples_y = pfenv.sample_poisson(min_distance=0.15)

    sampling.plot(samples_y, pfenv.domains, plot_f)


def plot_toy_poisson_path(pfenv: PotentialFlowEnv):
    y = np.array([[0.2, 0.1, 2.0], [0.197, 0.102, 3.5], [0.199, 0.098, 5.9]])
    _, samples_y = pfenv.resample_points_to_path(y)

    y = samples_y.reshape((-1, 3))

    plt.figure(figsize=(3.229, 2.8))

    if y.shape[1] == 2 or 3:
        plt.scatter(y[:, 0], y[:, 1], s=8,
                    c="lightgray", edgecolors='black')
        plt.scatter(y[20::25, 0], y[20::25, 1], s=16,
                    c="red", edgecolors='black')
    if y.shape[1] == 3:
        u = np.cos(y[:, 2])
        v = np.sin(y[:, 2])
        plt.quiver(y[:, 0], y[:, 1], u, v, color="black", zorder=-1)

    ax = plt.gca()
    ax.set_aspect("equal")

    plt.xlabel(r'$x(\mathrm{mm})$')
    plt.ylabel(r'$y(\mathrm{mm})$')

    plt.xticks([0.196, 0.198, 0.2, 0.202],
               ['196', '198', '200', '202'])
    plt.yticks([0.096, 0.098, 0.1, 0.102],
               ['96', '98', '100', '102'])

    plt.tight_layout()
    plt.show()


def plot_toy_path(pfenv: PotentialFlowEnv, duration=10):
    _, y = pfenv.sample_path(duration=duration)
    sampling.plot(y, pfenv.domains, plot_f, marker_size=.1)


def main():
    D_sensors = D
    dimensions = (2 * D, D)
    y_offset_v = Y_OFFSET

    sensors = SensorArray(N_SENSORS, (-0.4*D_sensors, 0.4*D_sensors))
    pfenv = PotentialFlowEnv(dimensions, y_offset_v, sensors, A, W_T)
    pfenv.show_env()

    plot_toy_poisson(pfenv)
    plot_toy_path(pfenv, 10)
    plot_toy_poisson_path(pfenv)


if __name__ == "__main__":
    main()
