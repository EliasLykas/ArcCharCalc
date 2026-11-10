import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_waveform(
    t_1: np.ndarray,
    v_1: np.ndarray,
    t_2: np.ndarray,
    v_2: np.ndarray,
    title: str,
    label_1: str = "1",
    label_2: str = "2",
) -> tuple[Figure, Axes]:
    """Plot two waveforms on the same axes, in different colours.

    Each waveform has its own time axis (µs), because the two are generally
    on different axes (e.g. withstand on t_w - tau_A, breakdown on t).
    Amplitudes are plotted as given (kV). The input arrays are not modified.
    For now the figure is shown here and the call blocks until the window is closed.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_1, v_1 , color="C0", label=label_1)
    ax.plot(t_2, v_2 , color="C3", label=label_2)
    ax.set_title(title)
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Amplitude (kV)")
    ax.grid(True)
    ax.legend()
    # shortcut: show() here only while testing; move it to the caller before batch runs (Phase 5) that save figures.
    plt.show()

    return fig, ax
