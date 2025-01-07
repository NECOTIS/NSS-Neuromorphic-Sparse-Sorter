"""
Created on 04.05.2024
@author: Alexis MELOT
@director: Sean WOOD
@co-directors: Fabien ALIBART, Pierre YGER, Yannick COFFINIER
"""
import time
import h5py as h5
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
from torch.nn import init
from tqdm import tqdm
from utils.tools import get_job_params, save_results, get_subdataset
from utils.build_dataset import init_dataset, MEADataset
from torch.utils.data import DataLoader
from model.Lca import LCA
import hdbscan
from utils.metrics import compute_accuracy
from plots.plot import lca_sorting, lca_output
from utils.tools import split_in_batches
from sklearn.linear_model import orthogonal_mp_gram

class aKSVD(object):
    def __init__(self, seed, natoms, nfeatures, max_iter=1, tol=1e-6,
                transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        natoms:
            Number of dictionary elements
        nfeatures:
            Number of features in input data
        nepochs:
            Maximum number of epochs to perform
        tol:
            tolerance for error
        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol
        self.natoms = natoms
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.D = self._initialize(nfeatures)

    def _update_dict(self, X, gamma):
        D = self.D
        for j in range(self.natoms):
            I = gamma[:, j] > 0
            if np.sum(I) == 0:
                continue
            
            D[j, :] = 0
            g = gamma[I, j].T
            r = X[I, :] - gamma[I, :].dot(D)
            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)
            D[j, :] = d
            gamma[I, j] = g.T
        return D, gamma

    def _initialize(self,nfeatures):
        """ Random initialization of dictionary."""
        np.random.seed(self.seed)
        # if min(X.shape) < self.n_components:
        D = np.random.randn(self.natoms, nfeatures)
        # else:
        #     u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
        #     D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, X):
        D = self.D
        gram = D.dot(D.T)
        Xy = D.dot(X.T)

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        return orthogonal_mp_gram(gram, Xy,
                    n_nonzero_coefs=n_nonzero_coefs).T

    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [nsamples, nfeatures]
        """
        for i in range(self.max_iter):
            gamma = self._transform(X)
            e = np.linalg.norm(X - gamma.dot(self.D))
            if e < self.tol:
                break
            self.D, gamma = self._update_dict(X, gamma)

        return self.D

    def transform(self, X):
        return self._transform(X)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.D, self._transform(X)
    
def main_ksvd(args):
    # init dataset
    infos, dataset = init_dataset(args.dataset_path, train_size=0.8, test_size=0.1) 
    dataset['train']['wv'] = dataset['train']['wv'] / torch.linalg.norm(dataset['train']['wv'], dim=1)[:,None]
    dataset['eval']['wv'] = dataset['eval']['wv'] / torch.linalg.norm(dataset['eval']['wv'], dim=1)[:,None]
    # split in batches
    batchs = split_in_batches(dataset, args.batch_size)
    xL = batchs['train']['wv']
    xE = batchs['eval']['wv']
    yE = batchs['eval']['raster']
    gt_raster_eval = np.vstack(yE).T

    # init ksvd
    ksvd = aKSVD(
            seed = 0,
            natoms = args.natoms,
            nfeatures = dataset['train']['wv'].shape[1],
            transform_n_nonzero_coefs = args.k) 

    # train
    a_train = []
    for bi in tqdm(range(len(xL))):
        _, a_bi = ksvd.fit_transform(xL[bi])
        a_train.append(a_bi)

    # eval
    a_eval = []
    for bi in tqdm(range(len(xE)), desc="evaluation"):
        a_eval_bi = ksvd.transform(xE[bi])
        a_eval.append(a_eval_bi)

    # clustering
    D = ksvd.D.T
    a_train = np.vstack(a_train)
    a_eval = np.vstack(a_eval)
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(a_train)
    c_eval = clusterer.fit_predict(a_eval)
    st_eval = gt_raster_eval.astype(int)[0]
    _, snr_acc = compute_accuracy(infos, c_eval, st_eval, gt_raster_eval, match_mode='best')
    
    return snr_acc, D, a_train, a_eval


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Script to train LCA on Synthetic MEA-dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/SYMEA_n48_nsd10_Neuropixels-24_0.h5",
        help="subset file path",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument(
        "--tau", type=float, default=0.02, help="Neurons' time constant."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Firing threshold."
    )
    parser.add_argument(
        "--natoms", type=int, default=280, help="Number of atoms in the dictionary."
    )
    parser.add_argument(
        "--iters", type=int, default=200, help="Number of LCA's iterations."
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument(
        "--beta", type=int, default=0, help="Fctor of the sparsity cost."
    )
    parser.add_argument(
        "--n_model",
        type=str,
        default="TDQ",
        help="Neuron model. Either 'IF','TDQ' or 'LIF'.",
    )
    parser.add_argument("--q", type=float, default=2**4-1, help="Quantization step.")
    return parser


def lca_iterates(lca, batch_loader, track=False, decay_lr=False):
    """LCA iterates over the dataset"""
    mse = []
    sparsity = []
    lasso = []
    s_sparsity = 0
    t_sparsity = 0
    coefs = []  # lca_outputs
    n = len(batch_loader)
    for id, (batch, _) in enumerate(tqdm(batch_loader)):
        if id == n//2 and track:
            lca.track = True
        if decay_lr:
            if id == n*0.75:
                lca.lr = lca.lr/2
            elif id == n*0.9:
                lca.lr = lca.lr/10
        lca(batch)
        coefs.append(lca.a)

        # metrics
        mse.append(lca.mse.cpu().numpy())
        sparsity.append(lca.sparsity.cpu().numpy())
        lasso.append(lca.lasso.cpu().numpy())
        t_sparsity += lca.t_sp.cpu().numpy()

    coefs = torch.vstack(coefs).cpu().numpy()
    return coefs, mse, sparsity, lasso, t_sparsity / n


def spike_sorting(infos, dataset, lca, train_loader, eval_loader, track=False, with_competition=True):
    """train/eval spike sorting pipeline : LCA + DBSCAN"""
    # Train
    lca.mode = "train"
    lca.with_competition = with_competition
    a_train, _, _, _, _, _ = lca_iterates(lca, train_loader)

    # Eval
    lca.mode = "eval"
    lca.with_competition = with_competition
    a_eval, mse, sparsity, loss, s_sp, t_sp = lca_iterates(
        lca, eval_loader, track=track
    )

    ## Clustering
    clusterer = hdbscan.HDBSCAN(
        core_dist_n_jobs=6,
    )
    clusterer.fit(a_train)
    c_eval = clusterer.fit_predict(a_eval)
    print("computing spike sorting accuracy...")
    gt_raster_eval = dataset["eval"]["raster"].cpu().numpy().T
    st_eval = gt_raster_eval.astype(int)[0]
    pred_raster, snr_acc = compute_accuracy(
        infos, c_eval, st_eval, gt_raster_eval, match_mode="best"
    )
    print("Mean spike sorting accuracy: %.2f" % (np.mean(snr_acc[:, 1])))
    return pred_raster, snr_acc, a_train, a_eval


def main(args, track=False, with_competition = True, hpc_run=False, seed=0) -> None:
    """Main function"""
    
    # datasets = [ds for ds in Path("dataset").iterdir()]
    # if hpc_run:
    #     job_id = get_job_params()
    #     args.dataset_path = datasets[job_id // 20]
    #     args.threshold = thresholds[job_id % 20]
    print(args)
    # build dataloader
    infos, dataset = init_dataset(args.dataset_path, train_size=0.8)
    # infos, dataset = get_subdataset(infos, dataset, nneurons=5, num_wv_per_neuron=150)
    train_loader = DataLoader(
        MEADataset(dataset["train"]["wv"], dataset["train"]["raster"]),
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=False,
    )
    eval_loader = DataLoader(
        MEADataset(dataset["eval"]["wv"], dataset["eval"]["raster"]),
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=False,
    )
    D = init.xavier_normal_(
        torch.zeros(
            (infos["input_size"], args.natoms),
            requires_grad=False,
            dtype=torch.float64,
            device="cpu",
        )
    )

    # run spike sorting pipeline
    lca = LCA(
        D=D,
        input_size=infos["input_size"],
        tau=args.tau,
        threshold=args.threshold,
        natoms=args.natoms,
        iters=args.iters,
        lr=args.lr,
        beta=args.beta,
        n_model=args.n_model,
        q=args.q,
        track=track,
        seed=seed,
    )
    infos["fs"] = 1e4
    tic = time.time()
    pred_raster, snr_acc, a_train, a_eval = spike_sorting(
        infos, dataset, lca, train_loader, eval_loader, with_competition = with_competition
    )
    dst_name = str(args.dataset_path).split("/")[-1]
    # save_results(dst_name, snr_acc, s_sp, t, args.threshold)
    print(f"exc time : {time.time() - tic}")
    # # plot
    raster_eval = dataset["eval"]["raster"].cpu().numpy().T
    # lca_sorting(infos, raster_eval, snr_acc, lca.D.cpu().numpy(), pred_raster)
    # if track:
    #     lca_output(np.vstack(lca.a_track).T)
    return snr_acc, lca.D.cpu().numpy(), a_train, a_eval, raster_eval


if __name__ == "__main__":
    model_name = "ksvd"
    logfile_name = 'ksvd_hp'
    parser = setup_args()
    args = parser.parse_args()
    args.dataset_path = 'data/SYMEA_n48_nsd10_Neuropixels-24_0.h5'
    args.k = 10 
    # snr_acc, D_learned, a_train, a_eval, gt_raster_eval = main(args, with_competition=False) #LCA
    snr_acc, D_learned, a_train, a_eval = main_ksvd(args)
    print(f"Mean spike sorting accuracy: {np.mean(snr_acc[:, 1])}")
    # plot snr accuracy
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100, tight_layout=True)
    # snr_acc = snr_acc[snr_acc[:, 1] > 0, :]  # keep acc>0
    ax.plot(snr_acc[:, 0], snr_acc[:, 1], color="k", lw=2)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs SNR")
    plt.show()
    # with h5.File(f"logs/saved_dict_coeffs_h5/{logfile_name}.h5", "a") as f:
    #     if str(args.dataset_path).split("/")[-1] in f.keys():
    #         g = f[str(args.dataset_path).split("/")[-1]]
    #     else:
    #         g = f.create_group(str(args.dataset_path).split("/")[-1])
    #     g1 = g.create_group(model_name)
    #     g1.create_dataset("snr_acc", data=np.array(snr_acc, dtype=np.float32))
    #     g1.create_dataset("D", data=np.array(D_learned))
    #     g1.create_dataset("a_train", data=np.array(a_train))
    #     g1.create_dataset("a_eval", data=np.array(a_eval))
    #     f.create_dataset("gt_raster_eval", data=np.array(gt_raster_eval))
    #     f.close()

    ## statistic on all datasets in data folder
    # datasets = [ds for ds in Path("data").iterdir() if str(ds).startswith("data/SYMEA_n48")]
    # for d in datasets:
    #     # args.dataset_path = d

    #     #LCA
    #     # snr_acc, D_learned, a_train, a_eval = main(args)
    #     # init dataset
    #     infos, dataset = init_dataset(d, train_size=0.8) 
    #     # split in batches
    #     batchs = split_in_batches(dataset, 64)
    #     yE = batchs['eval']['raster']
    #     gt_raster_eval = np.vstack(yE).T

    #     # # k-svd
    #     # args.k = 9
    #     # args.epochs = 3
    #     # args.iters = 318
    #     # snr_acc, D, a_train, a_eval = main_ksvd(args)

    #     #save coeffs, snr_acc and dict
    #     with h5.File(f"logs/saved_dict_coeffs_h5/alca_ksvd_dict_study_V3.h5", "a") as f:
    #         if str(d).split("/")[-1] in f.keys():
    #             g = f[str(d).split("/")[-1]]
    #         else:
    #             g = f.create_group(str(d).split("/")[-1])
    #         if 'gt_raster_eval' in g['lca'].keys():
    #             del g['lca']['gt_raster_eval']
    #         g['lca'].create_dataset("gt_raster_eval", data=np.array(dataset['eval']['raster'].T, dtype=np.int32))
    #         if 'gt_raster_eval' in g['ksvd'].keys():
    #             del g['ksvd']['gt_raster_eval']
    #         g['ksvd'].create_dataset("gt_raster_eval", data=np.array(gt_raster_eval, dtype=np.int32))
    #         # g1 = g.create_group(model_name)
    #         # g1.create_dataset("snr_acc", data=np.array(snr_acc))
    #         # g1.create_dataset("D", data=np.array(D_learned))
    #         # g1.create_dataset("a_train", data=np.array(a_train))
    #         # g1.create_dataset("a_eval", data=np.array(a_eval))
    #         f.close()
