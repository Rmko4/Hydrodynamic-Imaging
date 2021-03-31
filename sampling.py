import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc

rng = np.random.default_rng()

def plot(y):
    y = y.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Giving each class its own color and label
    ax.scatter(y[0], y[1], y[2])
    plt.show()

def uniform_sherical_annulus(size, center, inner_radius = 0, outer_radius = 1):
    n = center.shape[0]
    X = rng.normal(size=(size, n))
    s2 = np.sum(X**2,axis=1)

    n = 2
    x = np.array([np.linspace(-5, 5, 1000), np.linspace(-5, 5, 1000)]).transpose()
    s2sr = np.sum(x**2, axis=1)
    y = sc.gammainc(n/2, s2sr/2)
    ysr = sc.gammainc(n/2, s2sr/2)**(1/n)
    
    fsr = ysr/np.sqrt(s2sr)
    yx = x * np.tile(fsr.reshape(size,1),(1,n))
    plt.plot(x, y)
    plt.plot(x, ysr)
    plt.plot(x, fsr)
    plt.plot(x, yx)
    plt.show()
    
    keysl = sc.gammainc(n/2, s2/2)**(1/n)
    fr = outer_radius * keysl/np.sqrt(s2)
    y = X * np.tile(fr.reshape(size,1),(1,n))

    pass


def uniform_cube(domains):
    def random_uniform(x):
            return rng.uniform(x[0], x[1])

    return np.apply_along_axis(random_uniform, 1, domains)


def poisson_disk_sample(domains=np.array([(0, 1.0), (0, 1.0), (0, 2.0)]), r=0.1, k=30):
    n = domains.shape[0]
    cell_size = r / np.sqrt(n)

    def fit_cells(x): return np.ceil((x[1] - x[0]) / cell_size).astype(int)
    n_cells = np.apply_along_axis(fit_cells, 1, domains)

    grid = np.full(n_cells, -1)
    x_0 = uniform_cube(domains)
    samples = np.array([x_0])

    active_list = [0]
    while active_list:
        sel_i = rng.integers(len(active_list))
        index = active_list[sel_i]
        uniform_sherical_annulus(1000, samples[index], r, 2* r)
        pass
    pass


def main():

    poisson_disk_sample()


if __name__ == "__main__":
    main()
