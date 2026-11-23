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
    The figure is not shown here: the caller calls plt.show() (see `report.plot_run`).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_1, v_1 , color="C0", label=label_1)
    ax.plot(t_2, v_2 , color="C3", label=label_2)
    ax.set_title(title)
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    return fig, ax


def plot_events(
    t: np.ndarray,
    u: np.ndarray,
    i: np.ndarray,
    q: np.ndarray,
    e: np.ndarray,
    events: dict[str, float],
    title: str,
    window: tuple[float, float],
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot U and I (top), Q and E (bottom) around the events, like Figs 3.4-3.5 of Angeli-Knai (2022).

    All waveforms share t (μs, e.g. t_sync). events maps a name (e.g. "fj") to its time (μs):
    each gets a dashed vertical line and a dot on every curve. Only `window` (μs) is shown and
    each y axis is scaled to the data inside it. Not shown here, like plot_waveform.
    """
    lo, hi = window
    inside = (t >= lo) & (t <= hi)
    fig, (ax_u, ax_q) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    curves = [
        (ax_u, u, "C0", "U (kV)"),
        (ax_u.twinx(), i, "C3", "I_exc (A)"),  # twinx: second y axis on the right
        (ax_q, q, "C0", "Q_exc (μC)"),
        (ax_q.twinx(), e, "C3", "E_exc (mJ)"),
    ]
    for ax, x, color, label in curves:
        ax.plot(t, x, color=color)
        ax.set_ylabel(label, color=color)
        ax.tick_params(axis="y", colors=color)
        pad = 0.05 * (x[inside].max() - x[inside].min())
        ax.set_ylim(x[inside].min() - pad, x[inside].max() + pad)
        for te in events.values():
            ax.plot(te, np.interp(te, t, x), "o", color=color)
    for ax in (ax_u, ax_q):
        for name, te in events.items():
            ax.axvline(te, linestyle="--", color="gray")
            ax.annotate(name, (te, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom")
        ax.grid(True)
    ax_u.set_xlim(lo, hi)
    ax_u.set_title(title, pad=18)
    ax_q.set_xlabel("Time (μs)")
    return fig, (ax_u, ax_q)


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
    Returns (fig, ax_i, ax_u). Not shown here, like plot_waveform.
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
    return fig, ax_i, ax_u
