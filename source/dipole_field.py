import numpy as np
from utils.mpl_import import plt

def plot_dipole_field():
    slc = slice(-.5, .5, 100j)
    Y, X = np.mgrid[slc, slc]

    r = np.stack((X, Y), axis=-1)
    r_norm = np.expand_dims(np.linalg.norm(r, axis=-1), axis=-1)

    a = 1
    w = np.array([1, 0])

    w_dot_r = np.expand_dims(np.tensordot(w, r, axes=(0, 2)), axis=-1)
    v = a**3 / (2 * r_norm**3) * (-w + 3 * r * w_dot_r / (r_norm**2))
    U = v[:, :, 0]
    V = v[:, :, 1]

    speed = np.sqrt(U**2 + V**2)
    lw = 5*np.log(speed) / np.log(speed.max())

    # lines = np.linspace(0., 1., 20)
    # seed_points = np.array([lines, lines])
    # start_points=seed_points.T

    plt.streamplot(X, Y, U, V, density=1.7, color='grey',
                   linewidth=lw, arrowsize=.7, zorder=-1)
    plt.scatter([0], [0], color='black', s=100., zorder=1)
    plt.scatter([0], [0], color='grey', s=60., zorder=2)
    plt.quiver(0, 0, w[0], w[1], zorder=1, scale=12)

    ax = plt.gca()
    ax.set_aspect("equal")

    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    plt.tight_layout()
    plt.show()
    pass


if __name__ == "__main__":
    plot_dipole_field()
