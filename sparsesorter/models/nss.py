"""
Created on 16.04.2024
@author: Alexis MELOT
@director: Sean WOOD
@co-director: Fabien ALIBART, Pierre YGER, Yannick COFFINIER
University: U. Sherbrooke, U. Lille
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sparsesorter.models.lca import LCA
from torch.utils.data import DataLoader

###### NSS MODEL ######
class NSS(nn.Module):
    def __init__(
        self,
        input_size: int = 120,
        net_size: list = [120, 10],
        threshold: float | list = 0.1,
        gamma: float = 0.1,
        lr: float = 0.1,
        n_model = "TDQ",
        bit_width: int = 2,
        iters: int = 100,
        scale_factor: float = 0.8,
        fs: int = 10000,
        **_,
    ):
        super(NSS, self).__init__()
        #init NSS network layers
        self.lca1 = LCA(
            input_size=input_size,
            gamma=gamma,
            threshold=threshold,
            n_atoms=net_size[0],
            lr=lr,
            neuron_model=n_model,
            bit_width=bit_width,
        )
        self.lca2 = LCA(
            input_size=net_size[0],
            gamma=gamma,
            threshold=threshold,
            n_atoms=net_size[1],
            lr=lr,
            neuron_model=n_model,
            bit_width=bit_width,
            D_positive=True,
        )
        self.scale_factor = scale_factor
        self.iters = iters
        self.fs = fs



    def forward(self, input):
        self.lca1.init_vars(input.shape[0], self.iters)
        self.lca2.init_vars(input.shape[0], self.iters)

        # store exponential average of lca1.a and lca2.a computed on the last 20 iterations
        for it in range(self.iters):
            self.lca1.forward(input)
            self.lca1.n_spikes += torch.norm(self.lca1.a, p=0, dim=1)
            # if torch.any(self.lca1.a != 0): #propagate only if there is activity
            #     self.scale_factor = torch.max(self.lca1.a)
            scaled_output = torch.mul(self.lca1.a, 1/self.scale_factor)
            self.lca2.forward(scaled_output)
            self.lca2.n_spikes += torch.norm(self.lca2.a, p=0, dim=1)
            if it > self.iters-10:
                self.lca1.decoded_out += (self.lca1.a - self.lca1.decoded_out)/(10-(self.iters-it))
                self.lca2.decoded_out += (self.lca2.a - self.lca2.decoded_out)/(10-(self.iters-it))
            
            # store a to compute spatial and temporal sparsity
            self.lca1.a_t[it] = torch.t(self.lca1.a)
            self.lca2.a_t[it] = torch.t(self.lca2.a)
        
        # update wheights 
        self.lca1.update_dictionary(input)
        self.lca2.update_dictionary(scaled_output)

    def fit_transform(self, X:DataLoader)->torch.Tensor:
        """Spike sorting with NSS over the dataset.
        Input:
            X (DataLoader): Input data as a torch DataLoader
        Output:
            sparse_code (Tensor): Inferred LCA sparse codes
        """
        # training NSS
        nss_out = []
        n_spikes = []
        for _, (bi, ri) in enumerate(tqdm(X)):
            if int(ri[-1]) / self.fs > 120:
                self.lca1.lr, self.lca2.lr = 0.01, 0.01
                self.iters = 64
                n_spikes.append(self.lca1.n_spikes + self.lca2.n_spikes)
            self.forward(bi)
            nss_out.append(self.lca2.decoded_out.numpy())

        nss_out = np.concatenate(nss_out, axis=0)
        n_spikes = np.concatenate(n_spikes, axis=0)
        sorted_spikes = np.argmax(nss_out, axis=1).astype(int)

        return sorted_spikes, n_spikes





