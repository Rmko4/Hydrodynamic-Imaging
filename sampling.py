import numpy as np


def poisson_disk_sample(domains=np.array([(0, 1.0), (0, 1.0), (0, 2.0)]), r=0.1, k=30):
    n = domains.shape[0]
    cell_size = r / np.sqrt(n)

    def linspace2(x):
        return np.ceil((x[1] - x[0]) / cell_size)

    n_cells = np.apply_along_axis(linspace2, 1, domains)
    pass

def main():
    poisson_disk_sample()

if __name__ == "__main__":
    main()
