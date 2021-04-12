import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

def uniform_hyper_sherical_annulus(size, center, inner_radius, outer_radius):
    n = center.shape[0]
    x_bar = rng.normal(size=(size, n))
    s = np.sqrt(np.sum(x_bar**2, axis=1))
    x_bar = x_bar / s.reshape(-1, 1)  # Random direction (unit) vector

    u = rng.uniform(size=size)
    a_n = inner_radius ** n
    b_n = outer_radius ** n
    r = (u * (b_n - a_n) + a_n) ** (1/n)  # Density uniform radius

    y = center + r.reshape(-1, 1) * x_bar
    return y


def uniform_cube(domains):
    def random_uniform(x):
        return rng.uniform(x[0], x[1])

    return np.apply_along_axis(random_uniform, 1, domains)


def poisson_disk_sample(domains=np.array([[0, 1.0], [0, 1.0]]), r=0.05, k=30):
    n = domains.shape[0]
    cell_size = r / np.sqrt(n)
    r_sq = r * r

    n_cells = np.ceil((domains[:, 1] - domains[:, 0]) / cell_size).astype(int)

    def get_indices(point):
        return tuple(((point - domains[:, 0]) / cell_size).astype(int))

    def add_sample(point):
        samples.append(point)
        indices = get_indices(point)
        grid[indices] = len(samples) - 1
        active_list.append(grid[indices])
        pass

    def in_domain(point):
        return np.all((point >= domains[:, 0]) & (point < domains[:, 1]))

    def in_neighbourhood(point):
        indices = get_indices(point)
        cell_r = np.ceil(r/cell_size)
        grid_min = np.maximum(np.zeros(n), indices - cell_r).astype(int)
        grid_max = np.minimum(n_cells, indices + cell_r + 1).astype(int)

        iter = []
        for i in range(n):
            iter.append(slice(grid_min[i], grid_max[i]))
        sample_i = np.array(grid[tuple(iter)]).flatten()
        sample_i = sample_i[sample_i != -1]

        present_samples = np.take(samples, sample_i, axis=0)
        return np.any(np.sum((point - present_samples) ** 2, axis=1) < r_sq)

    grid = np.full(n_cells, -1)
    samples = []
    active_list = []

    x_0 = uniform_cube(domains)
    add_sample(x_0)

    while active_list:
        sel_i = rng.integers(len(active_list))
        index = active_list[sel_i]
        points = uniform_hyper_sherical_annulus(k, samples[index], r, 2 * r)
        new_point = False
        for point in points:
            if in_domain(point) and not in_neighbourhood(point):
                new_point = True
                add_sample(point)
        if not new_point:
            active_list.pop(sel_i)

    return np.array(samples)


def plot(X, axis_unit = "m"):
    if X.shape[1] == 2 or 3:
        plt.scatter(X[:, 0], X[:, 1], s=2, c="black")
    if X.shape[1] == 3:
        u = np.sin(X[:, 2])
        v = np.cos(X[:, 2])
        plt.quiver(X[:, 0], X[:, 1], u, v, color="grey")

    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    plt.grid(linestyle="--")

    plt.xlabel("x({})".format(axis_unit))
    plt.ylabel("y({})".format(axis_unit))
    plt.title("Sampled data points")

    plt.show()


def main():
    samples = poisson_disk_sample()
    print(samples.size)
    plot(samples)
    pass


if __name__ == "__main__":
    main()
