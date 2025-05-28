"""
Created on 16.04.2024
@author: Alexis MELOT
@director: Sean WOOD
@co-director: Fabien ALIBART, Pierre YGER, Yannick COFFINIER
University: U. Sherbrooke, U. Lille
"""
import torch
import torch.nn as nn
from torch.nn import init
from tqdm import tqdm
from torch.utils.data import DataLoader

DTYPE = torch.float32
DEVICE = torch.device("cpu")


###### NEURON MODELS ######
def v1_neuron(u, a, input_biais, Wr, gamma, threshold, a_positive=True):
    u_next = (
        torch.mul((1 - gamma), u)  # leak
        + torch.mul(gamma, input_biais)  # input biais
        - torch.mul(gamma, torch.mm(a, torch.t(Wr)))  # lateral inhibition
    )
    # apply rectified softshrink
    a_next = u_next*(u_next>threshold) - threshold
    
    if a_positive:#set to zeros for negative values of a 
        a_next[a_next<0] = 0

    return u_next, a_next


def v1_tdq_neuron(u, v_q, a, input_biais, Wr, gamma, threshold, bit_width=2):
    """Temporally Diffused Quantization
    of the v1 neuron output"""
    nsteps = 2**bit_width-1

    # v1 neuron
    u_v1, a_v1 = v1_neuron(u, a, input_biais, Wr, gamma, threshold)

    # tdq
    s = v_q + torch.mul(nsteps, a_v1)
    # k = torch.sign(s) * torch.floor(torch.abs(s))
    k = torch.floor(s) # seems to be better for q>1
    v_q = s - k
    a_q = torch.mul(k, 1 / nsteps)

    return u_v1, v_q, a_q


def lif_neuron(u, v, a, input_biais, Wr, gamma, threshold):
    a_next = torch.zeros_like(a)
    # u_next, u_th = v1_neuron(u, a, input_biais, Wr, gamma, 0.05)
    u_next = (
        u - torch.mul(gamma, u) #leak
        + torch.mul(gamma,input_biais) # input biais
        - torch.mul(gamma, torch.mm(a, torch.t(Wr))) # lateral inhibition
            )
    # u_th = F.softshrink(u_next, 0.05)
    v_next = (
        v - torch.mul(gamma, v) #leak
        + u_next  # membrane potential
    )
    
    # Fire and reset
    a_next = torch.where((v_next - threshold)>0, 1, 0)
    v_next -= a_next  # soft-reset
    # v_next[a_next>0] = 0 # hard-reset
    return u_next, v_next, a_next.to(dtype=DTYPE)

###### LCA MODEL ######
class LCA(nn.Module):
    def __init__(
        self,
        input_size: int,
        gamma: float,
        threshold: float = 0.03,
        n_atoms: int = 100,
        lr: float = 0.07,
        neuron_model: str = "TDQ",  #V1/TDQ/LIF
        bit_width: int = 2,
        D_positive: bool = False,
        seed:int=0,
        **_,
    ):
        super(LCA, self).__init__()
        self.gamma = gamma
        self.threshold = threshold
        self.natoms = n_atoms
        self.lr = lr
        self.n_model = neuron_model
        self.bit_width = bit_width
        self.D_positive = D_positive

        # losses
        self.mse = None
        self.l0_norm = None
        self.lasso = None
        self.ss =[]
        self.ts = []

        # init weights
        torch.manual_seed(seed)
        D = init.xavier_normal_(
            torch.zeros(
                (input_size, n_atoms),
                dtype=DTYPE,
                device=DEVICE,
            ),
            gain=1.44,
        )
        if self.D_positive:
            D = D + 1
        self.D = D / torch.norm(D, p=2, dim=0, keepdim=True)
        self.Wr = torch.mm(torch.t(self.D), self.D) - torch.eye(self.natoms).to(
            device=DEVICE, dtype=DTYPE
        )

    def update_dictionary(self, input, reduction="sum"):
        mse_loss = nn.MSELoss(reduction=reduction)
        recons_input = torch.mm(self.decoded_out, torch.t(self.D))
        self.mse = mse_loss(recons_input, input)
        l1_norm = torch.norm(self.decoded_out, p=1, dim=1)
        self.l0_norm = torch.norm(self.decoded_out, p=0, dim=1)
        self.lasso = self.mse + self.threshold * l1_norm

        # mse
        mse_gradient = -torch.mm(torch.t(input - recons_input), self.decoded_out)

        # update and normalize
        self.D -= self.lr * (mse_gradient + 0.03*torch.randn_like(self.D)) # random noise act as a anti-hebbian term
        if self.D_positive:
            self.D = torch.clamp(self.D , min=0)
        self.D /= torch.norm(self.D, p=2, dim=0, keepdim=True)
        self.Wr = torch.mm(torch.t(self.D), self.D) - torch.eye(self.natoms).to(
            device=DEVICE, dtype=DTYPE
        )   


    def init_vars(self, input_size, iters):
        self.u = torch.zeros(
            (input_size, self.natoms),
            dtype=DTYPE,
            device=DEVICE,
        )
        self.a = torch.zeros_like(self.u)
        self.v = torch.zeros_like(self.u)
        self.decoded_out = torch.zeros_like(self.u)
        self.n_spikes = torch.zeros(input_size, dtype=DTYPE, device=DEVICE)

        # store a to compute spatial and temporal sparsity
        self.a_t = torch.zeros((iters, self.natoms, input_size), dtype=DTYPE, device=DEVICE)
        

    def forward(self, input):
        input_biais = torch.mm(input, self.D)  # input biais
        if not hasattr(self, 'u'):
            self.init_vars(input.shape[0])           

        if self.n_model == "V1":
            self.u, self.a = v1_neuron(
                self.u,
                self.a,
                input_biais,
                self.Wr,
                self.gamma,
                self.threshold,
            )

        if self.n_model == "TDQ":
            self.u, self.v, self.a = v1_tdq_neuron(
                self.u,
                self.v,
                self.a,
                input_biais,
                self.Wr,
                self.gamma,
                self.threshold,
                self.bit_width,
            )

        if self.n_model == "LIF":
            self.u, self.v, self.a = lif_neuron(
                self.u,
                self.v,
                self.a,
                input_biais,
                self.Wr,
                self.gamma,
                self.threshold,
            )

    def fit_transform(self, X:DataLoader)->torch.Tensor:
        """LCA iterates over the dataset.
        Input:
            X (DataLoader): Input data as a torch DataLoader
        Output:
            sparse_code (Tensor): Inferred LCA sparse codes
        """
        sparse_code = []
        for _, (batch, _) in enumerate(X):
            self.forward(batch)
            sparse_code.append(self.a)
        sparse_code = torch.vstack(sparse_code).cpu().numpy()
        return sparse_code
