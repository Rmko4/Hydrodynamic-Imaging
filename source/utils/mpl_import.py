import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase

C_WIDTH = 3.229
PLOTS_PATH = 'plots/'
FIG_EXT = '.pdf'

params = {
    "text.usetex": True,
    "font.family": 'serif',
    'font.size': 10,
    "font.serif": ['Times'],
    'figure.figsize': (C_WIDTH, C_WIDTH),
    'figure.dpi': 200
}
plt.rcParams.update(params)
