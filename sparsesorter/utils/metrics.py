"""
Created on 16.04.2024
@author: Alexis MELOT
@director: Sean WOOD
@co-director: Fabien ALIBART, Pierre YGER, Yannick COFFINIER
University: U. Sherbrooke, U. Lille
"""
import numpy as np
import spikeinterface.full as si

class SortingMetrics(object):
    def __init__(
        self, labels, st_detected, gt_raster, fs, delta_time=2, match_mode="best"
    ):
        self.labels = labels
        self.st_detected = st_detected
        self.pred_raster = np.concatenate(
            [
                st_detected.reshape(1, -1),
                labels.reshape(1, -1),
            ],
            axis=0,
        ).astype(int)
        self.gtr = gt_raster
        self.fs = fs
        self.delta_time = delta_time
        self.match_mode = match_mode
        self.n_units = len(np.unique(gt_raster[1]))

    def get_sorting_perf(self, match_mode="best"):
        # compare predicted spike trains to gt
        si_gt_raster = si.NumpySorting.from_times_labels(
            [self.gtr[0]],
            [self.gtr[1]],
            self.fs,
        )
        si_pred_raster = si.NumpySorting.from_times_labels(
            [self.pred_raster[0]], [self.pred_raster[1]], self.fs
        )
        if match_mode == "hungarian":
            exhaustive_gt, compute_labels = True, True
        else:
            exhaustive_gt, compute_labels = False, False
        sorting_perf = si.compare_sorter_to_ground_truth(
            si_gt_raster,
            si_pred_raster,
            delta_time=self.delta_time,
            n_jobs=-1,
            match_mode=match_mode,
            match_score=0.2,
            exhaustive_gt=exhaustive_gt,
            compute_labels=compute_labels,
        )
        return sorting_perf

    def get_accuracy(self):
        sorting_perf = self.get_sorting_perf()
        self.accuracy = sorting_perf.get_performance()["accuracy"].values
        return self.accuracy

    def get_fscore(self):
        sorting_perf = self.get_sorting_perf(match_mode="hungarian")

        ## compute F-score
        matching_unit = sorting_perf.best_match_12.to_numpy().astype(int)
        conf_matrix = sorting_perf.get_confusion_matrix()
        self.fscore = np.zeros(self.n_units)
        for i in range(self.n_units):
            if matching_unit[i] == -1:  # no matching unit
                continue
            tp = conf_matrix[matching_unit[i]][i]
            fp = conf_matrix[matching_unit[i]].to_numpy()[-1]
            fn = conf_matrix["FN"].to_numpy()[i]
            self.fscore[i] = 2 * tp / (2 * tp + fp + fn)

        return self.fscore

#TODO: correct the function
def compute_fscore_evolution(sorted_spikes, dataset, packet_size:int=100):
    # compute fscore every packet of Ns spikes processed
    gtr = dataset["gt_raster"]
    nneurons = len(np.unique(gtr[1]))
    fs = dataset["fs"]
    spike_timing = dataset["raster"]
    spike_processed, fscore_nss = [], []
    for i in range(0, len(spike_timing), packet_size):
        if i + packet_size >= len(spike_timing):
            break
        mask_pred = (spike_timing >= spike_timing[i]) & (spike_timing < spike_timing[i + packet_size])
        mask_gtr = (gtr[0] >= spike_timing[i]) & (gtr[0] < spike_timing[i + packet_size])
        gtsort_comp = SortingMetrics(
            sorted_spikes[mask_pred],
            spike_timing[mask_pred],
            gtr[:, mask_gtr],
            fs,
            delta_time=2,
        )
        score = gtsort_comp.get_fscore()
        if not score.size > 0:
            continue
        else:
            spike_processed.append(i + packet_size)
            fscore_nss.append(score)
    if nneurons > 1:
        fscore_nss = np.array(fscore_nss).T
    else:
        fscore_nss = np.array(fscore_nss).reshape(nneurons, -1)
    spike_processed = np.array(spike_processed)
    return spike_processed, fscore_nss