"""
Created on 16.04.2024
@author: Alexis MELOT
@director: Sean WOOD
@co-director: Fabien ALIBART, Pierre YGER, Yannick COFFINIER
University: U. Sherbrooke, U. Lille
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
from metrics import SortingMetrics

figures_path = Path("../figures")
plt.style.use("seaborn-v0_8-paper")

def plot_nss_output(nss_out, dataset, rec_f, detection_th, time_range=[0, 10]):
    """
    Plot the NSS output and compare it with the ground truth raster.
    Parameters
    ----------
    nss_out : np.ndarray
        The output of the NSS algorithm, shape (nspikes, natoms_out).
    dataset : dict
        A dictionary containing the dataset information, including:
        - "raster": the raster of detected spikes.
        - "gt_raster": the ground truth raster.
        - "fs": the sampling frequency.
    rec_f : object
        The recording file object that provides access to the traces.
    detection_th : np.ndarray
        The detection thresholds for each channel, shape (nchan,).
    t_range : list, optional
        The time range to plot, specified as [start_time, end_time] in seconds.
        Default is [0, 10].    
    """
    
    sorted_spikes = np.argmax(nss_out, axis=1).astype(int) # select most active neuron
    gtsort_comp = SortingMetrics(
        sorted_spikes,
        dataset["raster"],
        dataset["gt_raster"],
        dataset["fs"],
        delta_time=1,
    )
    sorting_perf = gtsort_comp.get_sorting_perf(match_mode="hungarian")
    best_match_12 = sorting_perf.best_match_12
    natoms_out = nss_out.shape[1]

    fs = rec_f.get_sampling_frequency()
    nneurons = np.unique(dataset["gt_raster"][1]).shape[0]
    nchan = rec_f.get_num_channels()

    _ = plt.figure(figsize=(6, 7), dpi=150)
    t = np.arange(time_range[0], time_range[1], 1 / fs)
    peaks_train = dataset["raster"]
    mask_trange = (peaks_train >= time_range[0] * fs) & (peaks_train < time_range[1] * fs)
    peaks = peaks_train[mask_trange]
    trace = rec_f.get_traces()[int(time_range[0] * fs) : int(time_range[1] * fs), :]
    min_trace, max_trace = np.min(trace)-10, np.max(trace)+10

    gs = gridspec.GridSpec(6, 1, height_ratios=[0.1,0.1,0.1,0.1, 0.15, 0.4])
    # create a subplot of 4 rows and 1 column with gs[0:3]
    ax03 = [plt.subplot(gs[i]) for i in range(4)]
    ax1 = plt.subplot(gs[4])
    ax2 = plt.subplot(gs[5])
    for ch in range(nchan):
        trace_ch = rec_f.get_traces()[int(time_range[0] * fs) : int(time_range[1] * fs), ch]
        ax03[ch].plot(t, trace_ch, c="k", alpha=0.5)  # trace
        ax03[ch].axhline(-detection_th[ch], c="k", linestyle="--")  # detection threshold
        ax03[ch].spines[["bottom", "top", "right"]].set_visible(False)
        ax03[ch].set_xticks([])
        ax03[ch].set_ylim(min_trace, max_trace)
        ax03[ch].set_ylabel(f"Ch{ch+1}")
        for p in peaks:
            win_width = 3 * fs // 1000
            trace_window = rec_f.get_traces()[p - int(0.4 * win_width) : p + int(0.6 * win_width),:]
            p -= int(time_range[0] * fs)
            max_chan = np.argmax(np.max(np.abs(trace_window), axis=0))
            if max_chan == ch:
                t_window = t[p - int(0.4 * win_width) : p + int(0.6 * win_width)]
                ax03[ch].plot(t_window, trace_window[:,ch], c='k')

    peaks = peaks - int(time_range[0] * fs)
    # plot gt_raster_train on the same time range
    gtr_train = dataset["gt_raster"]
    gtr_train = gtr_train[
        :, (gtr_train[0] >= time_range[0] * fs) & (gtr_train[0] < time_range[1] * fs)
    ]
    c_unit = plt.cm.Set1(np.linspace(0, 1, 9))
    for i in range(nneurons):
        idx = np.where(gtr_train[1] == i)[0]
        ax1.vlines(gtr_train[0][idx] / fs, i - 0.4, i + 0.4, color=c_unit[i], lw=0.8)
    ax1.set_ylabel("Ground Truth")
    ax1.spines[["bottom", "top", "right"]].set_visible(False)
    ax1.set_yticks(np.arange(0, nneurons, 1))
    # ax1.set_yticklabels([])
    ax1.set_xticks([])

    # plot pred_raster on the same time range
    c_out_nss = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, out_i in enumerate(nss_out[mask_trange]):
        peak_i = peaks[i]
        atom_active = len(out_i) - np.argmax(out_i)
        ax2.vlines(peak_i / fs, atom_active - 0.4, atom_active + 0.4, color="k", lw=0.8)
    ax2.set_ylabel("Inferred Raster - NSS")
    ax2.spines[["top", "right"]].set_visible(False)
    # set y-ticks at every 1 unit but label at every 2 units
    ax2.set_yticks(np.arange(1, natoms_out+1, 1))
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", which="major")
    ax2.set_yticks(np.arange(1, natoms_out+1,2), minor=True)
    ax2.set_yticklabels(np.arange(1, natoms_out+1,2), minor=True)
    ax2.tick_params(axis="y", which="minor", labelsize=8)
    ax2.set_xticks(np.arange(t[0], t[-1] + 1, 1) - t[0])
    ax2.set_xticklabels(np.arange(t[0], t[-1] + 1, 1, dtype=int))
    ax2.set_xlabel("Time (s)")

    # draw a rectangle around the vlines of the atom 0
    for ni in range(nneurons):
        pos_y = nss_out.shape[1] - best_match_12[ni] - 0.4
        len_x = time_range[1] - time_range[0]
        ax2.add_patch(
            plt.Rectangle(
                (0, pos_y), len_x + 0.1, 0.8, fill=False, edgecolor=c_unit[ni], lw=1
            )
        )
    plt.savefig(figures_path / "fig3.svg")
    plt.show()