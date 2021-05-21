import numpy as np

def reduce_polyfit(u_bar_s, true_idx, alpha=3.25e-9, beta=3.5e-10, gamma=6e-9, n_poly=3):
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

        u_red_s.append(u_red)

    return np.array(u_red_s)
