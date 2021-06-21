import numpy as np
from utils.mpl_import import plt
from utils.arraylist import ArrayList
from sklearn.metrics import pairwise_distances

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


def uniform_hyper_cube(domains):
    return rng.uniform(domains[:, 0], domains[:, 1])


def poisson_disc_sample(domains=np.array([[0., 1.0], [0., 1.0]]), r=0.05, k=30, wrap_domains=[]):
    n = domains.shape[0]
    cell_size = r / np.sqrt(n)
    r_sq = r * r

    n_cells = np.ceil((domains[:, 1] - domains[:, 0]) / cell_size).astype(int)

    def get_indices(point):
        return tuple(((point - domains[:, 0]) / cell_size).astype(int))

    def add_sample(point):
        samples.append(point)
        indices = get_indices(point)
        grid[indices] = samples.size - 1
        active_list.append(grid[indices])
        pass

    def in_domain(point):
        return np.all((point >= domains[:, 0]) & (point < domains[:, 1]))

    def in_neighbourhood(point):
        indices = get_indices(point)
        cell_r = np.ceil(r/cell_size)
        grid_min = np.maximum(np.zeros(n), indices - cell_r).astype(int)
        grid_max = np.minimum(n_cells, indices + cell_r + 1).astype(int)

        u_grid = grid

        iter = []
        for i in range(n):
            if i in wrap_domains:
                left = indices[i] - int(cell_r)
                right = indices[i] + int(cell_r) + 1
                if left < 0:
                    u_grid = np.roll(grid, -left, axis=i)
                    right -= left
                    left = 0
                elif right > n_cells[i]:
                    shift = n_cells[i]-right
                    u_grid = np.roll(grid, shift, axis=i)
                    left += shift
                    right = n_cells[i]
                iter.append(slice(left, right))
            else:    
                iter.append(slice(grid_min[i], grid_max[i]))
        sample_i = u_grid[tuple(iter)].flatten()
        sample_i = sample_i[sample_i != -1]

        present_samples = samples.data.take(sample_i, axis=0)
        return np.any(np.sum((point - present_samples) ** 2, axis=1) < r_sq)

    grid = np.full(n_cells, -1)
    samples = ArrayList(shape=(None, n), dtype=np.float32)
    active_list = []

    x_0 = uniform_hyper_cube(domains)
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
                break
        if not new_point:
            active_list.pop(sel_i)

    return samples.finalize()


def sample_path_on_pos(p_0, step_distance=.5,
                       max_turn_angle=np.pi/4, n_fwd=4, n_bwd=15,
                       inner_sampling_factor=1):
    def forward(p, distance):
        x = p[0] + distance * np.cos(p[2])
        y = p[1] + distance * np.sin(p[2])
        return np.array([x, y, p[2]])
    
    n_paths = np.shape(p_0)[0]
    path_len = n_fwd + n_bwd + 1

    paths = np.zeros((n_paths, path_len, 3))

    for i in range(n_paths):
        x = p_0[i]
        paths[i, n_bwd] = x
        for j in range(n_bwd + 1, path_len):
            for _ in range(inner_sampling_factor):
                x = forward(x, step_distance)
                x[2] = (x[2] + rng.uniform(-max_turn_angle,
                                           max_turn_angle)) % (2*np.pi)
            paths[i, j] = x
        x = p_0[i]
        for j in reversed(range(0, n_bwd)):
            for _ in range(inner_sampling_factor):
                x[2] = (x[2] + rng.uniform(-max_turn_angle,
                                           max_turn_angle)) % (2*np.pi)
                x = forward(x, -step_distance)
            paths[i, j] = x
    return paths


def sample_path_2D(domains=np.array([[0., 1.0], [0., 1.0]]), step_distance=.05,
                   max_turn_angle=np.pi/4, circum_radius=None, n_samples=100,
                   inner_sampling_factor=1, mode='rotate'):

    

    if circum_radius is None:
        n_reg_poly = 2 * np.pi / max_turn_angle
        circum_radius = step_distance / (2 * np.sin(np.pi/n_reg_poly))
    else:
        if step_distance > 2 * circum_radius:
            value = step_distance / 2
            raise ValueError(f"circum_radius should be >= {value}")
        n_reg_poly = np.pi / np.arcsin(step_distance / (2 * circum_radius))
        max_turn_angle = 2 * np.pi / n_reg_poly

    turn_radius = circum_radius + .5 * step_distance
    d_radius = 3 * circum_radius

    if mode == "rotate":
        reduction = np.tile([d_radius, -d_radius], (2, 1))
        initial_s_domains = domains[0:2] + reduction
    else:
        initial_s_domains = domains[0:2]

    initial_s_domains = np.concatenate([initial_s_domains, [[0, 2*np.pi]]])

    def forward(p):
        x = p[0] + step_distance * np.cos(p[2])
        y = p[1] + step_distance * np.sin(p[2])
        return np.array([x, y, p[2]])

    def rotate(x):
        phi_addition = None
        if x[0] < domains[0, 0] + d_radius and x[1] < domains[1, 0] + d_radius:
            if x[0] < domains[0, 0] + turn_radius and x[1] > domains[1, 0] + turn_radius and np.pi < x[2] <= 1.25*np.pi:
                phi_addition = -1.5*np.pi
            elif x[0] > domains[0, 0] + turn_radius and x[1] < domains[1, 0] + turn_radius and 1.25*np.pi <= x[2] < 1.5*np.pi:
                phi_addition = -np.pi
            else:
                phi_addition = -0.25*np.pi
        elif x[0] < domains[0, 0] + d_radius and x[1] > domains[1, 1] - d_radius:
            if x[0] < domains[0, 0] + turn_radius and x[1] < domains[1, 1] - turn_radius and 0.75*np.pi <= x[2] < np.pi:
                phi_addition = -0.5*np.pi
            elif x[0] > domains[0, 0] + turn_radius and x[1] > domains[1, 1] - turn_radius and 0.5*np.pi < x[2] <= 0.75*np.pi:
                phi_addition = -np.pi
            else:
                phi_addition = -1.75*np.pi
        elif x[0] > domains[0, 1] - d_radius and x[1] < domains[1, 0] + d_radius:
            if x[0] < domains[0, 1] - turn_radius and x[1] < domains[1, 0] + turn_radius and 1.5*np.pi < x[2] <= 1.75*np.pi:
                phi_addition = -0.5*np.pi
            elif x[0] > domains[0, 1] - turn_radius and x[1] > domains[1, 0] + turn_radius and 1.75*np.pi <= x[2] < 2*np.pi:
                phi_addition = np.pi
            else:
                phi_addition = -0.75*np.pi
        elif x[0] > domains[0, 1] - d_radius and x[1] > domains[1, 1] - d_radius:
            if x[0] < domains[0, 1] - turn_radius and x[1] > domains[1, 1] - turn_radius and 0.25*np.pi <= x[2] < 0.5*np.pi:
                phi_addition = -1.5*np.pi
            elif x[0] > domains[0, 1] - turn_radius and x[1] < domains[1, 1] - turn_radius and 0 < x[2] <= 0.25*np.pi:
                phi_addition = -np.pi
            else:
                phi_addition = -1.25*np.pi
        elif x[0] < domains[0, 0] + turn_radius:
            phi_addition = 0
        elif x[0] > domains[0, 1] - turn_radius:
            phi_addition = np.pi
        elif x[1] < domains[1, 0] + turn_radius:
            phi_addition = 1.5*np.pi
        elif x[1] > domains[1, 1] - turn_radius:
            phi_addition = 0.5*np.pi

        if phi_addition is None:
            # phi = rng.normal(0, max_turn_angle)
            # phi = (x[2] + np.clip(phi, -max_turn_angle, max_turn_angle)) % (2*np.pi)
            phi = (x[2] + rng.uniform(-max_turn_angle,
                                      max_turn_angle)) % (2*np.pi)
        else:
            norm_x2 = (x[2] + phi_addition) % (2*np.pi)

            if 0 <= norm_x2 <= np.pi:
                phi = norm_x2 - max_turn_angle
            else:
                phi = norm_x2 + max_turn_angle
            phi = 0 if phi >= 2 * np.pi or phi < 0 else phi
            phi = (phi - phi_addition) % (2*np.pi)

        x[2] = phi
        return x

    def reflect(x):
        reflect = False
        if x[0] < domains[0, 0]:
            x[0] = 2 * domains[0, 0] - x[0]
            x[2] = (np.pi - x[2]) % (2*np.pi)
            reflect = True
        elif x[0] > domains[0, 1]:
            x[0] = 2 * domains[0, 1] - x[0]
            x[2] = (np.pi - x[2]) % (2*np.pi)
            reflect = True
        if x[1] < domains[1, 0]:
            x[1] = 2 * domains[1, 0] - x[1]
            x[2] = -x[2] % (2*np.pi)
            reflect = True
        elif x[1] > domains[1, 1]:
            x[1] = 2 * domains[1, 1] - x[1]
            x[2] = -x[2] % (2*np.pi)
            reflect = True
        if reflect is False:
            x[2] = (x[2] + rng.uniform(-max_turn_angle,
                                       max_turn_angle)) % (2*np.pi)
        return x

    boundary_f = reflect if mode == 'reflect' else rotate

    path = []
    x = uniform_hyper_cube(initial_s_domains)
    path.append(x)

    n_path_iters = inner_sampling_factor * n_samples - 1
    for i in range(n_path_iters):
        x = forward(x)
        x = boundary_f(x)

        if i % inner_sampling_factor == inner_sampling_factor - 1:
            path.append(x)

    return np.array(path)


def plot(X, domains=None, plot_f=None, marker_size=10):
    plt_c = plt

    if callable(plot_f):
        plot_f(plt_c)

    if X.shape[1] == 2 or 3:
        plt.scatter(X[:, 0], X[:, 1], s=marker_size,
                    c="grey", edgecolors='black')
    if X.shape[1] == 3:
        u = np.cos(X[:, 2])
        v = np.sin(X[:, 2])
        plt.quiver(X[:, 0], X[:, 1], u, v, color="black", zorder=-1)

    if domains is not None:
        plt.hlines(domains[1, :], domains[0, 0],
                   domains[0, 1], colors='grey', linestyles='--')
        plt.vlines(domains[0, :], domains[1, 0],
                   domains[1, 1], colors='grey', linestyles='--')

    ax = plt.gca()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


def print_sample_metrics(samples, feature_subsets=None):
    print_mean_min_distance(samples)
    for lb, ub in feature_subsets:
        print_mean_min_distance(samples[:, lb:ub])


def print_mean_min_distance(samples):
    pwd = pairwise_distances(samples)
    np.fill_diagonal(pwd, np.inf)
    pwd_min = pwd.min(0)
    print(np.mean(pwd_min))


def main():
    # samples_path = sample_path_2D(mode='rotate')
    # plot(samples_path)

    samples = poisson_disc_sample(
        np.array([[0., 1.0], [0., 1.0], [0., 1.0]]), r=0.08)
    # print(len(samples))
    # plot(samples)
    pass


if __name__ == "__main__":
    main()
