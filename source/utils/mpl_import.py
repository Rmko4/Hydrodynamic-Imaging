import matplotlib.pyplot as plt
import matplotlib.patches as patches

C_WIDTH = 3.229

params = {
    "text.usetex": False,
    "font.family": 'serif',
    'font.size': 10,
    "font.serif": ['Times'],
    'figure.figsize': (C_WIDTH, C_WIDTH),
    'figure.dpi': 200
}
plt.rcParams.update(params)
