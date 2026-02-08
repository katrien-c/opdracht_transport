import matplotlib as mpl
import matplotlib.pyplot as plt

 

def rescale_rgb(color):
              return [rgb / 255 for rgb in color]

 

DARKRED = rescale_rgb([192, 0, 0])
RED = rescale_rgb([255, 0, 0])
ORANGE = rescale_rgb([255, 192, 0])
YELLOW = rescale_rgb([255, 255, 0])
GREEN = rescale_rgb([146, 208, 80])
DARKGREEN = rescale_rgb([0, 176, 80])
LIGHTBLUE = rescale_rgb([0, 176, 240])
BLUE = rescale_rgb([0, 112, 192])
DARKBLUE = rescale_rgb([0, 32, 96])
PURPLE = rescale_rgb([112, 48, 160])
LIGHTGRAY = rescale_rgb([224, 225, 227])


def standard_style():
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['figure.figsize'] = (8, 4)
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['axes.prop_cycle'] = plt.cycler(color = [DARKBLUE,
                                                             DARKGREEN,
                                                             LIGHTBLUE,
                                                             ORANGE,
                                                             LIGHTGRAY,
                                                             PURPLE,
                                                             DARKRED,
                                                             RED])
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['font.size'] = 16
    return