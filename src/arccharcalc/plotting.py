import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_waveform(t: np.ndarray, v: np.ndarray, title: str) -> tuple[Figure, Axes]:
    """Plot a waveform: time on the x axis, amplitude (scaled to kV) on the y axis.

    The input arrays are not modified. The caller decides whether to show or
    save the figure.
    """
    fig, ax = plt.subplots()
    ax.plot(t, v / 1e3)
    ax.set_title(title)
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Amplitude (kV)")
    ax.grid(True)
    return fig, ax
