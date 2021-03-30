import numpy as np


def poisson_disk_sample(domains=np.array([(0, 1.0), (0, 1.0)]), r=0.1, k=30):
    n = domains.size
    cell_size = r / np.sqrt(n)
    
