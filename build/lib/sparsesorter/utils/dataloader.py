"""
Created on 16.04.2024
@author: Alexis MELOT
@director: Sean WOOD
@co-director: Fabien ALIBART, Pierre YGER, Yannick COFFINIER
University: U. Sherbrooke, U. Lille
"""
import torch
import h5py as h5
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DTYPE = torch.float32
DEVICE = torch.device("cpu")

class SSDataset(Dataset):
    def __init__(self, wv, st, normalize=True):
        if normalize:
            waveforms = wv / torch.linalg.norm(wv, dim=1).reshape(-1, 1)
        else:
            waveforms = wv
        self.xs = waveforms.to(dtype=DTYPE, device=DEVICE, non_blocking=True)
        self.ys = st.to(dtype=DTYPE, device=DEVICE, non_blocking=True)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]
    
def init_dataloader(X, y, batch_size, normalize=True):
    if type(X) != torch.Tensor and type(y) != torch.Tensor:
        X = torch.tensor(X, requires_grad=False, dtype=torch.float32, device="cpu")
        y = torch.tensor(y, requires_grad=False, dtype=torch.float32, device="cpu")
    else:
        X = X.to("cpu")
        y = y.to("cpu")
    dataset = SSDataset(X, y, normalize=normalize)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, pin_memory=False
    )
    return dataloader

def build_dataloader(ds, tmax=240, fs=10000, batch_size=16):
    data_file = ds
    with h5.File(data_file, "r") as f:
        wvs = np.array(f["wvs"][:], dtype=np.float32)
        gt_raster = np.array(f["gt_raster"][:], dtype=np.int32)
        peaks_idx = np.array(f["peaks_idx"][:], dtype=np.int32)
        try:
            snr = np.array(f["snr"][:], dtype=np.float32)
        except:
            snr = np.array(f["snr"], dtype=np.float32)
    f.close()
    # normalize waveforms with l2-norm
    l2_norm = np.linalg.norm(wvs, ord=2, axis=1)
    if np.sum(l2_norm < 1e-6) > 0:
        print("Warning: some waveforms are null")
    wvs = wvs / np.linalg.norm(wvs, ord=2, axis=1)[:, None]
    # filter and keep only wvs which peaks_idx are below tmax
    mask_detected = peaks_idx < tmax * fs
    mask_gt = gt_raster[0] < tmax * fs
    wvs, peaks_idx = wvs[mask_detected], peaks_idx[mask_detected]
    gt_raster = gt_raster[:, mask_gt]
    dataset = {"wvs": wvs, "gt_raster": gt_raster, "raster": peaks_idx, "snr": snr, "fs": fs}
    dataloader = init_dataloader(wvs, peaks_idx, batch_size, normalize=False)
    return dataset, dataloader


def compute_detection_performance(dataset, delta_time=1, fs=10000):
    """Compute predicted label for each peak according to their closest ground truth spike """
    gtr = dataset["gt_raster"]
    snr = dataset["snr"]
    _, counts = np.unique(gtr[1], return_counts=True)
    tp, fn, fp = np.zeros(len(snr)), np.zeros(len(snr)), 0
    well_detected_spikes, not_detected_gt_spikes = [], []
    peaks_idx, peaks_idx_copy = dataset["raster"], np.copy(dataset["raster"])
    labels_peaks = -1 * np.ones(len(peaks_idx))
    
    for i in range(gtr.shape[1]):
        idx = np.where(
            np.abs(peaks_idx_copy - gtr[0, i]) <= delta_time * fs / 1000
        )  # search for a spike in a 1ms range
        if idx[0].size > 0:
            tp[gtr[1, i]] += 1
            well_detected_spikes.append(i)
            idx_closest = np.argmin(np.abs(peaks_idx_copy - gtr[0, i]))
            labels_peaks[np.where(peaks_idx == peaks_idx_copy[idx_closest])] = gtr[1, i]
            peaks_idx_copy = np.delete(peaks_idx_copy, idx_closest)
        else:
            fn[gtr[1, i]] += 1
            not_detected_gt_spikes.append(i)
    fp = len(peaks_idx) - len(well_detected_spikes)
    precision = tp / counts
    recall = tp / (tp + fn)
    fprate = fp / (fp + tp)

    res_detection = {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "well_detected_spikes": well_detected_spikes,
        "not_detected_gt_spikes": not_detected_gt_spikes,
        "labels_peaks": labels_peaks,
        "fp_rate": fprate,
    }
    return res_detection


