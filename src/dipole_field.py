import numpy as np
import matplotlib.pyplot as plt


def plot_dipole_field():
    slc = slice(-1, 1, 100j)
    Y, X = np.mgrid[slc, slc]

    r = np.stack((Y, X), axis=-1)
    r_norm = np.expand_dims(np.linalg.norm(r, axis=-1), axis=-1)

    a = 1
    w = np.array([0, 1])

    w_dot_r = np.expand_dims(np.tensordot(w, r, axes=(0, 2)), axis=-1)
    v = a**3 / (2 * r_norm**3) * (-w + 3 * r * w_dot_r / (r_norm**2))
    V = v[:, :, 0]
    U = v[:, :, 1]

    speed = np.sqrt(U**2 + V**2)
    lw = 5*np.log(speed) / np.log(speed.max())

    # lines = np.linspace(0., 1., 20)
    # seed_points = np.array([lines, lines])
    # start_points=seed_points.T
    plt.streamplot(X, Y, U, V, density=1, color="grey",
                   linewidth=lw)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.show()
    pass


if __name__ == "__main__":
    plot_dipole_field()
