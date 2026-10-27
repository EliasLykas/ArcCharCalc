import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

def plot_waveform(t, v, title):
    """Plot a waveform given time and amplitude arrays."""
    fig, ax = plt.subplots()
    ax.plot(t, v/1e3)  
    ax.set_title(title)
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Amplitude (kV)")
    ax.grid(True)
    plt.show()