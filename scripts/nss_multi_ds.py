import os
import pandas as pd
import pickle
import h5py as h5
import numpy as np
from tqdm import tqdm
from utils.metrics import GTSortingComparison
from utils.build_dataset import init_dataset_online
from model.Lca import LCA1iter, NSS_online
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

fs = 10000
batch_size = 16
datasets = [32, 62, 149, 652]
ntrials = 1 
nneurons = 4 
nchan = 4

def main():
    seed = 0
    for s in range(ntrials):
        for i, ds in enumerate(datasets):
            print(f"Dataset: {ds} - trial: {s}")
            data_file = f"data/tetrode/tetrode{ds}_n4_static.h5"
            with h5.File(data_file, "r") as f:
                wvs = np.array(f["wvs"][:], dtype=np.float32)
                gt_raster = np.array(f["gt_raster"][:], dtype=np.int32)
                snr = np.array(f["snr"][:], dtype=np.float32)
                peaks_idx = np.array(f["peaks_idx"][:], dtype=np.int32)
                # wvs_gt = np.array(f["wvs_gt"][:], dtype=np.float32)
            f.close()

            # normalize waveforms with l2-norm
            l2_norm = np.linalg.norm(wvs, ord=2, axis=1)
            if np.sum(l2_norm < 1e-6) > 0:
                print("Warning: some waveforms are null")
            wvs = wvs / np.linalg.norm(wvs, ord=2, axis=1)[:, None]

            dataset, dataloaders = init_dataset_online(
                wvs,
                peaks_idx,
                gt_raster,
                eval_size=0.1,
                test_size=0.3, #can be greater because the validation has not been used
                batch_size=batch_size,
            )
            params_nss = {
                "n_atoms1": 240,
                "n_atoms2": 10,
                "D1_positive": False,
                "D2_positive": True,
                "th1": 0.06,
                "th2": 0.06,
                "fs": fs,
                "tau": 2e-3,
                "iters": 150,
                "lr": 0.08,
                "n_model": "TDQ",
                "q": 2**2 - 1,
                "seed": seed,
            }
            params_nss["gamma"] = 1 / params_nss["fs"] / params_nss["tau"]

            ## init lca1
            lca1 = LCA1iter(
                input_size=wvs.shape[1],
                gamma=params_nss["gamma"],
                threshold=params_nss["th1"],
                n_atoms=params_nss["n_atoms1"],
                lr=params_nss["lr"],
                neuron_model=params_nss["n_model"],
                q=params_nss["q"],
                D_positive=params_nss["D1_positive"],
                seed=params_nss["seed"],
            )
            lca2 = LCA1iter(
                input_size=params_nss["n_atoms1"],
                gamma=params_nss["gamma"],
                threshold=params_nss["th2"],
                n_atoms=params_nss["n_atoms2"],
                lr=params_nss["lr"],
                neuron_model=params_nss["n_model"],
                q=params_nss["q"],
                D_positive=params_nss["D2_positive"],
                seed=params_nss["seed"],
            )
            nss = NSS_online(lca1, lca2, params_nss["iters"], scale_factor=0.8)

            # training lca
            for _, (bi, _) in enumerate(tqdm(dataloaders["train"])):
                nss(bi)

            # evaluate
            nss.lca1.mode = "eval"
            nss.lca2.mode = "eval"
            nss_out = []
            for _, (bi, _) in enumerate(tqdm(dataloaders["test"])):
                nss(bi)
                nss_out.append(nss.lca2.decoded_out.numpy())
            nss_out = np.concatenate(nss_out, axis=0)
            label = np.argmax(nss_out, axis=1).astype(int)
            gtsort_comp = GTSortingComparison(
                label,
                dataset["test"]["raster"],
                dataset["test"]["gt_raster"],
                fs,
                delta_time=2,
            )
            fscore_nss = gtsort_comp.get_fscore()
            print(f"F1-score nss: {fscore_nss.mean()*100:.3f}")
            snr_fscore_nss = np.stack([snr, fscore_nss], axis=1).astype(np.float32)
            snr_fscore_nss = pd.DataFrame(snr_fscore_nss, columns=["snr", "fscore"])

            # PCA
            pcs_test = np.zeros((dataset["test"]["wv"].shape[0], nchan, 3))
            for chi in range(nchan):
                wvs_train = dataset["train"]["wv"]
                wvs_test = dataset["test"]["wv"]
                wvs_train = wvs_train.reshape(wvs_train.shape[0], nchan, -1)
                wvs_test = wvs_test.reshape(wvs_test.shape[0], nchan, -1)
                pca = PCA(n_components=3, random_state=seed)
                pca.fit(wvs_train[:, chi, :])
                pcs_test[:, chi] = pca.transform(wvs_test[:, chi, :])
            pcs_test = pcs_test.reshape(-1, nchan * 3)
            kmeans = KMeans(n_clusters=nneurons, random_state=seed).fit(pcs_test)
            label_pk = kmeans.labels_
            gtsort_comp = GTSortingComparison(
                label_pk,
                dataset["test"]["raster"],
                dataset["test"]["gt_raster"],
                fs,
                delta_time=2,
            )
            fscore_pk = gtsort_comp.get_fscore()
            print(f"F1-score pca: {fscore_pk.mean()*100:.3f}")
            snr_fscore_pk = np.stack([snr, fscore_pk], axis=1).astype(np.float32)
            snr_fscore_pk = pd.DataFrame(snr_fscore_pk, columns=["snr", "fscore"])

            # save results
            save_results("logs/tetrode_nss_vs_pca.pkl", f"tetrode{ds}", snr_fscore_nss, s)
            save_results("logs/tetrode_nss_vs_pca.pkl", f"tetrode{ds}", snr_fscore_pk, s)

        seed += 1
        


def save_results(file_path:str, dst_name: str, snr_acc: np.ndarray, trial: int):
    if os.path.exists(file_path):
        previous_perf = pd.read_pickle(file_path)
    else:
        previous_perf = pd.DataFrame(
            columns=["model", "dataset_name", "snr", "fscore", "trial"]
        )
    new_perf = pd.DataFrame(snr_acc, columns=["snr", "accuracy"], dtype=np.float32)
    new_perf.insert(0, "model", "nss")
    new_perf.insert(1, "dataset_name", dst_name)
    # 2 and 3 are reserved for snr and fscore
    new_perf.insert(4, "trial", trial)

    res_df = pd.concat([previous_perf, new_perf], axis=0)
    res_df.to_pickle(file_path)


if __name__ == "__main__":
    main()
    