import numpy as np
from multiprocessing import Pool
from sampling import rng


def reduce_polyfit(u_bar_s, true_idx, alpha=5.2e-9, beta=5.6e-10, gamma=9.6e-9, n_poly=3):
    degs = range(n_poly + 1)

    win_len = np.shape(u_bar_s)[1]
    signal_len = np.shape(u_bar_s)[2]

    t = np.linspace(0, win_len - 1, win_len)

    u_red_s = []

    for u_bar in u_bar_s:
        u_red = np.empty(signal_len)

        for i in range(signal_len):
            residual = 1
            for deg in degs:
                res = np.polyfit(t, u_bar[:, i], deg, full=True)
                delta_residual = residual - res[1][0]   
                if deg == 0 or (residual > alpha and delta_residual > beta):
                    p = np.poly1d(res[0])
                residual = res[1][0]

            if residual > gamma:
                u_red[i] = u_bar[true_idx, i]
            else:
                u_red[i] = p(t[true_idx])
            pass
        u_red_s.append(u_red)

    return np.array(u_red_s)

def _MSE_model(path_u, u_true, alpha=5.2e-9, beta=5.6e-10, gamma=9.6e-9):
    samples_u = reduce_polyfit(path_u, -5, alpha, beta, gamma)
    err = np.mean((samples_u - u_true) ** 2)
    return err

def search_best_model(path_u, u_true, iters=100):
    param_bounds = [[1e-9, 1e-8], [1e-11, 1e-10], [8e-9, 3e-8]]
    alpha = rng.uniform(*param_bounds[0], size=iters)
    beta = rng.uniform(*param_bounds[1], size=iters)
    gamma = rng.uniform(*param_bounds[2], size=iters)
    path_u = iters * [path_u]
    u_true = iters * [u_true]
    iter_args = zip(path_u, u_true, alpha, beta, gamma)

    with Pool() as p:
        errors = p.starmap(_MSE_model, iter_args)
    
    idx = np.argsort(errors)
    for i in idx[0:10]:
        print(alpha[i])
        print(beta[i])
        print(gamma[i])
        print(errors[i])
        print()
    print()
