import numpy as np
from utils.mpl_import import plt, patches


def plot_mlp():
    zeros = 4 * [0.]
    ones = 4 * [1.]
    one_halfs = 4 * [1.5]
    twos = 3 * [2.]
    threes = 4 * [3.]

    y_0 = [-0.5, 0., 0.3, 0.6]
    plt.scatter(zeros, y_0, s=500, c='white', edgecolors='black', zorder=1)
    plt.scatter(ones, y_0, s=500, c='white', edgecolors='black', zorder=1)

    y_1 = [-0.5, 0., 0.3, 0.6]

    y_2 = [-0.65, -0.3, 0., 0.3]
    plt.scatter(twos, y_2[1:], s=500, c='white',
                edgecolors='black', zorder=1)

    y_3 = [-0.5, -0.2, 0.3, 0.6]
    plt.scatter(threes, y_3, s=500, c='white', edgecolors='black', zorder=1)

    y_s = [-0.2, -0.25, -0.3]
    plt.scatter(zeros[0:3], y_s, s=10, c='black', marker='.', zorder=1)
    plt.scatter(ones[0:3], y_s, s=10, c='black', marker='.', zorder=1)
    plt.scatter(threes[0:3], [0, 0.05, 0.1], s=10,
                c='black', marker='.', zorder=1)

    for i in y_0:
        for j in y_0[0:3]:
            plt.plot([0., 1.], [i, j], c='black', zorder=0)

    for i in y_0:
        for j in y_1[0:3]:
            plt.plot([1., 1.5], [i, j], c='black', zorder=0)

    for i in y_1:
        for j in y_2[1:]:
            plt.plot([1.5, 2.], [i, j], c='black', zorder=0)

    for i in y_2[1:]:
        plt.plot([2., 2.5], [i, 0.], c='black', zorder=0, ls='--')
    for i in y_3:
        plt.plot([2.5, 3.0], [0., i], c='black', zorder=0, ls='--')

    y_bar_names = [r'$\hat{\mathbf{y}}$',
                   r'$\hat{\varphi}$', r'$\hat{d}$', r'$\hat{b}$']

    for i in range(len(y_2)):
        plt.annotate(y_bar_names[i], (2., y_2[i]), ha='center', va='center')

    plt.annotate(r'$\mathbf{u}$', (0., -0.65), ha='center', va='center')
    plt.annotate(r'Hidden layer 1', (1., -0.65), ha='center', va='center')
    plt.annotate(r'$\hat{\mathbf{u}}$', (3., -0.65), ha='center', va='center')

    names_0 = [r'$\mu_{yP}$', r'$\mu_{y1}$', r'$\mu_{x1}$', r'$1.0$']
    names_1 = [r'$x_{L^1}^1$', r'$x_{2}^1$', r'$x_{1}^1$', r'$1.0$']
    names_2 = [r'$\hat{\mu}_{yP}$', r'$\hat{\mu}_{xP}$', r'$\hat{\mu}_{y1}$', r'$\hat{\mu}_{x1}$']
    names_n = [names_0, names_1, names_2]

    pos_x = [0., 1., 3.]
    pos_y = [y_0, y_0, y_3]

    for names, x, y in zip(names_n, pos_x, pos_y):
        for j in range(len(y_0)):
            plt.annotate(names[j], (x, y[j]), ha='center', va='center')

    plt.annotate('Bias units', (0.5, 0.65), ha='center', va='center')

    ellipse = patches.Ellipse(
        (1.5, 0.), 0.3, 1.5, fc='white', ec=None, zorder=1)

    r_h = 0.25
    r_w = 0.3
    rect = patches.Rectangle(
        (2.5-r_w/2, 0.-r_h/2), r_w, r_h, fc='white', ec='black', zorder=1)

    plt.annotate(r'$\tilde{v}_*$',
                 (2.5, 0), ha='center', va='center')

    ax = plt.gca()
    ax.add_patch(ellipse)
    ax.add_patch(rect)

    for i in y_0:
        plt.scatter([1.45, 1.5, 1.55], 3 * [i], s=10,
                    c='black', marker='.', zorder=2)
    plt.scatter(one_halfs[0:3], y_s, s=10, c='black', marker='.', zorder=2)

    r_h = 1.40
    r_w = 2.46
    rect = patches.Rectangle(
        (-.23, -.70), r_w, r_h, fc='none', ec='lightgrey', ls='--', zorder=1)
    ax.add_patch(rect)

    r_h = 1.46
    r_w = 1.46
    rect = patches.Rectangle(
        (1.77, -.73), r_w, r_h, fc='none', ec='darkgrey', ls='--', zorder=1)
    ax.add_patch(rect)

    plt.xlim(-.25, 3.25)
    plt.ylim(-.75, 0.75)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_mlp()
