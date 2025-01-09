"""
Created on 16.04.2024
@author: Alexis MELOT
@director: Sean WOOD
@co-director: Fabien ALIBART, Pierre YGER, Yannick COFFINIER
University: U. Sherbrooke, U. Lille
"""
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si
from sparsesorter.models.nss import NSS
from sparsesorter.utils.metrics import compute_fscore_evolution, SortingMetrics
from sparsesorter.utils.dataloader import build_dataloader
from pathlib import Path

DATA_PATH = Path("data")
DATASET = "TS1"
BIT_WIDTH = 1

def load_data(ds_name="TS1"):
    ds_file = DATA_PATH / f"{ds_name}.h5"
    dataset, dataloader = build_dataloader(ds_file,180)
    return dataset, dataloader


def init_nss(dataset, net_size=[120, 10], threshold=0.03, gamma=0.05, lr=0.07):
    nss = NSS(
        input_size=dataset["wvs"].shape[1],
        net_size=net_size,
        threshold=threshold,
        gamma=gamma,
        lr=lr,
        bit_width=BIT_WIDTH,
    )
    return nss


def run_nss(ds_name="TS1"):
    dataset, dataloader = load_data(ds_name)
    nss = init_nss(dataset)
    nss_out, _ = nss.fit_transform(dataloader)
    sorted_spikes = np.argmax(nss_out, axis=1).astype(int)  # select most active neuron
        
    sorted_spikes = np.argmax(nss_out, axis=1).astype(int)  # select most active neuron
    packet_size = 400
    _, f1score = compute_fscore_evolution(
        sorted_spikes, dataset, packet_size
    )
    id_snr = np.argsort(dataset["snr"])
    snr = dataset["snr"][id_snr]
    f1score = f1score[id_snr]
    return snr, f1score[:,5:].mean(axis=1) # return mean F1-score for each gt-neuron when NSS is stable

if __name__ == "__main__":
    print(f"Running NSS-{BIT_WIDTH}bit on {DATASET}...")
    snr, f1score = run_nss(DATASET)
    print(f"NSS sorting results: \n SNR:{snr.round(1)} \n  F1-Score:{f1score.round(2)}")  