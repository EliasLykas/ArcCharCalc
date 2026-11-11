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
    ylabel: str = "Amplitude (kV)",
) -> tuple[Figure, Axes]:
    """Plot two waveforms on the same axes, in different colours.

    Each waveform has its own time axis (µs), because the two are generally
    on different axes (e.g. withstand on t_w - tau_A, breakdown on t).
    Amplitudes are plotted as given, in the unit named by ylabel (default kV). The input arrays are not modified.
    For now the figure is shown here and the call blocks until the window is closed.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_1, v_1 , color="C0", label=label_1)
    ax.plot(t_2, v_2 , color="C3", label=label_2)
    ax.set_title(title)
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    # shortcut: show() here only while testing; move it to the caller before batch runs (Phase 5) that save figures.
    plt.show()

    return fig, ax


def plot_current_voltage(
    t: np.ndarray,
    i: np.ndarray,
    u: np.ndarray,
    title: str,
    i_label: str = "Current",
    u_label: str = "Voltage",
) -> tuple[Figure, Axes, Axes]:
    """Plot current (left axis, A) and voltage (right axis, kV) on one time axis (µs).

    Both waveforms share t (e.g. t_sync); each y axis has its own scale.
    Returns (fig, ax_i, ax_u). Shown here, like plot_waveform.
    """
    fig, ax_i = plt.subplots(figsize=(10, 5))
    ax_u = ax_i.twinx()  # second y axis on the right, same x axis
    line_i = ax_i.plot(t, i, color="C0", label=i_label)
    line_u = ax_u.plot(t, u, color="C3", label=u_label)
    ax_i.set_xlabel("Time (μs)")
    ax_i.set_ylabel(f"{i_label} (A)", color="C0")
    ax_u.set_ylabel(f"{u_label} (kV)", color="C3")
    # Put zero at the same height on both axes: give both the same ratio |bottom| / top.
    # Assumes both waves are mainly positive (polarity already applied), so top > 0.
    ratio = max(-ax.get_ylim()[0] / ax.get_ylim()[1] for ax in (ax_i, ax_u))
    for ax in (ax_i, ax_u):
        ax.set_ylim(-ratio * ax.get_ylim()[1], ax.get_ylim()[1])
    ax_i.tick_params(axis="y", colors="C0")
    ax_u.tick_params(axis="y", colors="C3")
    ax_i.set_title(title)
    ax_i.grid(True)
    lines = line_i + line_u  # one legend for the lines of both axes
    ax_i.legend(lines, [line.get_label() for line in lines])
    # shortcut: show() here only while testing; move it to the caller before batch runs (Phase 5).
    plt.show()

    return fig, ax_i, ax_u
