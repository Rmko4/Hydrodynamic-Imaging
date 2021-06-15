import numpy as np
from utils.mpl_import import plt


def plot_velocity_profile(n=4):
    rho = np.linspace(-2., 2., 1000)

    def Psi_e(rho):
        rho_sq = np.square(rho)
        return (2 * rho_sq - 1) / np.power(1 + rho_sq, 2.5)

    def Psi_o(rho):
        rho_sq = np.square(rho)
        return (-3 * rho) / np.power(1 + rho_sq, 2.5)

    def Psi_n(rho):
        rho_sq = np.square(rho)
        return (2 - rho_sq) / np.power(1 + rho_sq, 2.5)

    phi = np.linspace(0, 2*np.pi, n, endpoint=False)

    cos_phi = np.cos(phi).reshape(-1, 1)
    sin_phi = np.sin(phi).reshape(-1, 1)

    v_x = (Psi_e(rho) * cos_phi + Psi_o(rho) * sin_phi)
    v_y = (Psi_o(rho) * cos_phi + Psi_n(rho) * sin_phi)/2

    c1 = plt.get_cmap('Greens')(np.linspace(0.3, 0.7, n))
    c2 = plt.get_cmap('Reds')(np.linspace(0.3, 0.7, n))

    def plot_profile(v, c, ylab=r'$v$', annotation_pos=None):
        plt.figure(figsize=(3.229, 2.1))

        for i in range(n):
            plt.plot(rho, v[i], lw=1, c=c[i])

        plt.xlabel(r'$\rho$')
        plt.ylabel(ylab)
        plt.grid(ls='dashed', c='lightgrey')

        if annotation_pos:
            plt.annotate(r'$0^{\circ}$', annotation_pos[0],
                         ha='center', va='center')
            plt.annotate(r'$90^{\circ}$', annotation_pos[1],
                         ha='center', va='center')
            plt.annotate(r'$180^{\circ}$', annotation_pos[2],
                         ha='center', va='center')
            plt.annotate(r'$270^{\circ}$', annotation_pos[3],
                         ha='center', va='center')

        plt.tight_layout()
        plt.show()

    annotation_pos_x = [(0.02, -0.84), (-0.535, 0.67), (0.05, 0.77), (0.56, 0.67)]
    annotation_pos_y = [(-0.46, 0.265), (0.01, 0.81), (0.51, 0.265), (0.03, -0.825)]

    plot_profile(v_x, c1, r'$\tilde{v}_x$', annotation_pos_x)
    plot_profile(v_y, c2, r'$\tilde{v}_y$', annotation_pos_y)


if __name__ == "__main__":
    plot_velocity_profile()
