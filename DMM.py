# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Deep Markov Model in Pyro adapted for continuous, stimulus-driven, sparsely
observed C. elegans GCaMP neural fluorescence time series.

Reference:
[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""

import argparse
import logging
import time
from os.path import exists
from typing import Optional, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import ClippedAdam
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from pyro_svi_c302 import DEFAULT_DATA_PATH
from utils import AtlasLoader, RandiTrialDataset, responseGenerator


def reverse_sequences_pad(sequences: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each sequence along time up to its valid length (batch_first=True)."""
    out = torch.zeros_like(sequences)
    for i, length in enumerate(seq_lengths.tolist()):
        l = int(length)
        if l > 0:
            out[i, :l] = torch.flip(sequences[i, :l], dims=[0])
    return out


class Emitter(nn.Module):
    """Parameterizes continuous Gaussian observation likelihood p(x_t | z_t)."""

    def __init__(self, input_dim: int, z_dim: int, emission_dim: int):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_loc = nn.Linear(emission_dim, input_dim)
        self.lin_hidden_to_scale = nn.Linear(emission_dim, input_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        loc = self.lin_hidden_to_loc(h2)
        scale = self.softplus(self.lin_hidden_to_scale(h2)) + 1e-2
        return loc, scale


class GatedTransition(nn.Module):
    """Parameterizes stimulus-conditioned Gaussian latent transition p(z_t | z_{t-1}, u_t)."""

    def __init__(self, z_dim: int, transition_dim: int, stim_dim: int = 0):
        super().__init__()
        in_dim = z_dim + stim_dim
        self.lin_gate_z_to_hidden = nn.Linear(in_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(in_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(
        self, z_t_1: torch.Tensor, u_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = z_t_1 if u_t is None else torch.cat([z_t_1, u_t], dim=-1)
        _gate = self.relu(self.lin_gate_z_to_hidden(inp))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(inp))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean))) + 1e-3
        return loc, scale


class Combiner(nn.Module):
    """Parameterizes variational posterior q(z_t | z_{t-1}, x_{t:T}, u_{t:T})."""

    def __init__(self, z_dim: int, rnn_dim: int):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1: torch.Tensor, h_rnn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        loc = self.lin_hidden_to_loc(h_combined)
        scale = self.softplus(self.lin_hidden_to_scale(h_combined)) + 1e-3
        return loc, scale


class DMM(nn.Module):
    """Deep Markov Model with continuous Gaussian emissions and missing-neuron masking."""

    def __init__(
        self,
        input_dim: int = 300,
        z_dim: int = 64,
        emission_dim: int = 128,
        transition_dim: int = 128,
        rnn_dim: int = 256,
        num_layers: int = 1,
        rnn_dropout_rate: float = 0.0,
        num_iafs: int = 0,
        iaf_dim: int = 64,
        use_stim: bool = True,
        use_cuda: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_stim = use_stim
        stim_dim = input_dim if use_stim else 0

        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim, stim_dim=stim_dim)
        self.combiner = Combiner(z_dim, rnn_dim)

        rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
        rnn_in_dim = input_dim * 2 if use_stim else input_dim
        self.rnn = nn.RNN(
            input_size=rnn_in_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=rnn_dropout_rate,
        )

        self.iafs = [
            affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)
        ]
        self.iafs_modules = nn.ModuleList(self.iafs)

        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(num_layers, 1, rnn_dim))

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def model(
        self,
        mini_batch: torch.Tensor,
        mini_batch_reversed: torch.Tensor,
        mini_batch_mask: torch.Tensor,
        mini_batch_seq_lengths: torch.Tensor,
        annealing_factor: float = 1.0,
        stim_batch: Optional[torch.Tensor] = None,
    ):
        T_max = mini_batch.size(1)
        pyro.module("dmm", self)
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))
        time_mask = mini_batch_mask.any(dim=-1, keepdim=True)  # (B, T, 1)

        with pyro.plate("z_minibatch", len(mini_batch)):
            for t in pyro.markov(range(1, T_max + 1)):
                u_t = None if (not self.use_stim or stim_batch is None) else stim_batch[:, t - 1, :]
                z_loc, z_scale = self.trans(z_prev, u_t)
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        f"z_{t}",
                        dist.Normal(z_loc, z_scale).mask(time_mask[:, t - 1, :]).to_event(1),
                    )
                obs_loc, obs_scale = self.emitter(z_t)
                pyro.sample(
                    f"obs_x_{t}",
                    dist.Normal(obs_loc, obs_scale)
                    .mask(mini_batch_mask[:, t - 1, :])
                    .to_event(1),
                    obs=mini_batch[:, t - 1, :],
                )
                z_prev = z_t

    def guide(
        self,
        mini_batch: torch.Tensor,
        mini_batch_reversed: torch.Tensor,
        mini_batch_mask: torch.Tensor,
        mini_batch_seq_lengths: torch.Tensor,
        annealing_factor: float = 1.0,
        stim_batch: Optional[torch.Tensor] = None,
    ):
        T_max = mini_batch.size(1)
        pyro.module("dmm", self)

        h_0_contig = self.h_0.expand(
            self.rnn.num_layers, mini_batch.size(0), self.rnn.hidden_size
        ).contiguous()
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        rnn_output = reverse_sequences_pad(rnn_output, mini_batch_seq_lengths)

        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))
        time_mask = mini_batch_mask.any(dim=-1)  # (B, T)

        with pyro.plate("z_minibatch", len(mini_batch)):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                else:
                    z_dist = dist.Normal(z_loc, z_scale).to_event(1)
                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(f"z_{t}", z_dist.mask(time_mask[:, t - 1]))
                z_prev = z_t


def _prepare_batch(
    indices: torch.Tensor,
    y_all: torch.Tensor,
    stim_all: torch.Tensor,
    mask_all: torch.Tensor,
    seq_lengths: torch.Tensor,
    use_stim: bool,
    device: torch.device,
):
    mini_batch = y_all[indices].to(device)
    stim_batch = stim_all[indices].to(device)
    mini_mask = mask_all[indices].to(device)
    lens = seq_lengths[indices].to(device)
    rnn_in = torch.cat([mini_batch, stim_batch], dim=-1) if use_stim else mini_batch
    mini_rev = reverse_sequences_pad(rnn_in, lens)
    return mini_batch, mini_rev, mini_mask, lens, stim_batch


def main(args) -> None:
    logging.basicConfig(
        level=logging.DEBUG, format="%(message)s", filename=args.log, filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    logging.info(args)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    if args.source == "randi":
        if args.dataset_cache and exists(args.dataset_cache):
            ds = RandiTrialDataset.load(args.dataset_cache)
        else:
            ds = AtlasLoader(
                path=args.data_path,
                strain=args.strain,
                norm_mode=args.norm_mode,
                max_recordings=args.max_recordings,
            ).build_trial_dataset(pre_vols=args.pre_vols, post_vols=args.post_vols)
            if args.dataset_cache:
                ds.save(args.dataset_cache)
        x_all, y_all, mask_all = ds.stim_inputs, ds.y, ds.obs_mask
    else:
        x_all, y_all, mask_all = responseGenerator(
            folder="./wormfunconn/atlas/", strain=args.strain, size=100
        ).Dataset(return_mask=True)

    idx = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=42)
    x_train, y_train, m_train = x_all[train_idx], y_all[train_idx], mask_all[train_idx]
    x_test, y_test, m_test = x_all[test_idx], y_all[test_idx], mask_all[test_idx]

    N_train_data = len(y_train)
    T_seq = y_train.shape[1]
    training_seq_lengths = torch.full((N_train_data,), T_seq, dtype=torch.long)
    val_seq_lengths = torch.full((len(y_test),), T_seq, dtype=torch.long)
    N_train_obs = float(m_train.sum().item())
    N_mini_batches = max(
        1,
        int(N_train_data / args.mini_batch_size + int(N_train_data % args.mini_batch_size > 0)),
    )

    logging.info(
        "N_train_data: %d  seq_len: %d  N_mini_batches: %d",
        N_train_data,
        T_seq,
        N_mini_batches,
    )

    val_batch, val_batch_reversed, val_batch_mask, val_lengths_dev, val_stim = _prepare_batch(
        torch.arange(len(y_test)), y_test, x_test, m_test, val_seq_lengths, args.use_stim, device
    )

    dmm = DMM(
        input_dim=y_train.shape[2],
        rnn_dropout_rate=args.rnn_dropout_rate,
        num_iafs=args.num_iafs,
        iaf_dim=args.iaf_dim,
        use_stim=args.use_stim,
        use_cuda=(device.type == "cuda"),
    )

    adam = ClippedAdam(
        {
            "lr": args.learning_rate,
            "betas": (args.beta1, args.beta2),
            "clip_norm": args.clip_norm,
            "lrd": args.lr_decay,
            "weight_decay": args.weight_decay,
        }
    )
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)

    def save_checkpoint():
        if args.save_model:
            torch.save(dmm.state_dict(), args.save_model)
        if args.save_opt:
            adam.save(args.save_opt)

    def load_checkpoint():
        assert exists(args.load_opt) and exists(args.load_model), "--load-model and/or --load-opt misspecified"
        dmm.load_state_dict(torch.load(args.load_model, map_location=device))
        adam.load(args.load_opt)

    def process_minibatch(epoch, which_mini_batch, shuffled_indices):
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
            min_af = args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * (
                float(which_mini_batch + epoch * N_mini_batches + 1)
                / float(args.annealing_epochs * N_mini_batches)
            )
        else:
            annealing_factor = 1.0

        start = which_mini_batch * args.mini_batch_size
        end = min((which_mini_batch + 1) * args.mini_batch_size, N_train_data)
        mb_idx = shuffled_indices[start:end]
        mb, mb_rev, mb_mask, mb_lens, mb_stim = _prepare_batch(
            mb_idx, y_train, x_train, m_train, training_seq_lengths, args.use_stim, device
        )
        return svi.step(mb, mb_rev, mb_mask, mb_lens, annealing_factor, mb_stim)

    def do_evaluation() -> float:
        dmm.rnn.eval()
        n_val_obs = max(1.0, float(val_batch_mask.sum().item()))
        val_nll = svi.evaluate_loss(
            val_batch, val_batch_reversed, val_batch_mask, val_lengths_dev, 1.0, val_stim
        ) / n_val_obs
        dmm.rnn.train()
        return float(val_nll)

    if args.load_opt and args.load_model:
        load_checkpoint()

    val_test_frequency = 5
    times = [time.time()]
    for epoch in range(args.num_epochs):
        if args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
            save_checkpoint()

        epoch_nll = 0.0
        shuffled_indices = torch.randperm(N_train_data)
        for which_mini_batch in range(N_mini_batches):
            epoch_nll += process_minibatch(epoch, which_mini_batch, shuffled_indices)

        times.append(time.time())
        logging.info(
            "[training epoch %04d]  %.4f \t(dt = %.3f sec)",
            epoch,
            epoch_nll / max(1.0, N_train_obs),
            times[-1] - times[-2],
        )
        if val_test_frequency > 0 and (epoch + 1) % val_test_frequency == 0:
            val_nll = do_evaluation()
            logging.info("[val epoch %04d]  %.4f", epoch, val_nll)
    if args.save_model or args.save_opt:
        save_checkpoint()


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.9"), f"Expected Pyro 1.9.x, found {pyro.__version__}"

    parser = argparse.ArgumentParser(description="Train DMM on C. elegans GCaMP traces")
    parser.add_argument("--source", choices=["randi", "synthetic"], default="randi")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--dataset-cache", type=str, default=None)
    parser.add_argument("--strain", type=str, default="wild-type")
    parser.add_argument("--norm-mode", choices=["dff_safe", "dff_strict", "robust_z"], default="dff_safe")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--pre-vols", type=int, default=10)
    parser.add_argument("--post-vols", type=int, default=50)
    parser.add_argument("--no-stim", dest="use_stim", action="store_false")
    parser.add_argument("-n", "--num-epochs", type=int, default=20)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001)
    parser.add_argument("-b1", "--beta1", type=float, default=0.95)
    parser.add_argument("-b2", "--beta2", type=float, default=0.999)
    parser.add_argument("-cn", "--clip-norm", type=float, default=10.0)
    parser.add_argument("-lrd", "--lr-decay", type=float, default=0.99996)
    parser.add_argument("-wd", "--weight-decay", type=float, default=0.0)
    parser.add_argument("-mbs", "--mini-batch-size", type=int, default=16)
    parser.add_argument("-ae", "--annealing-epochs", type=int, default=10)
    parser.add_argument("-maf", "--minimum-annealing-factor", type=float, default=0.2)
    parser.add_argument("-rdr", "--rnn-dropout-rate", type=float, default=0.1)
    parser.add_argument("-iafs", "--num-iafs", type=int, default=0)
    parser.add_argument("-id", "--iaf-dim", type=int, default=64)
    parser.add_argument("-cf", "--checkpoint-freq", type=int, default=0)
    parser.add_argument("-lopt", "--load-opt", type=str, default="")
    parser.add_argument("-lmod", "--load-model", type=str, default="")
    parser.add_argument("-sopt", "--save-opt", type=str, default="")
    parser.add_argument("-smod", "--save-model", type=str, default="")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("-l", "--log", type=str, default="dmm.log")
    args = parser.parse_args()

    main(args)
