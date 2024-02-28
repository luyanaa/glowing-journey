import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.tuner import Tuner

import os
import time
import pyro, pyro.optim
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive, SMCFilter
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.distributions import Empirical
import pyro.distributions as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from torchinfo import summary

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from utils import AtlasLoader, responseGenerator, PsuedoDataset
import c302
from soft_dtw_cuda import SoftDTW

# Stochastic Variational Inference, Monte Carlo Markov Chain and Sequential Monte Carlo Filter.
epochs_run = 0

def _save_snapshot(snapshot_path, model, epoch):
    snapshot = {
            "MODEL_STATE": model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
    torch.save(snapshot, snapshot_path)
    print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")   


# We define an ELBO loss, a PyTorch optimizer, and a training step in our PyroLightningModule.
# Note that we are using a PyTorch optimizer instead of a Pyro optimizer and
# we are using ``training_step`` instead of Pyro's SVI machinery.
class PyroLightningModule(pl.LightningModule):
    def __init__(self, loss_fn: pyro.infer.elbo.ELBOModule, lr: float):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = loss_fn.model
        self.guide = loss_fn.guide
        self.lr = lr
        self.predictive = pyro.infer.Predictive(
            self.model, guide=self.guide, num_samples=1
        )
        self.acc = SoftDTW(False)

    def forward(self, *args):
        return self.predictive(*args)

    def training_step(self, batch, batch_idx):
        """Training step for Pyro training."""
        loss = self.loss_fn(*batch)
        external_train, voltage_clamp, y = batch
        y_hat = self.model(external_train, voltage_clamp)
        acc = self.acc(y_hat, y)
        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        loss = self.loss_fn(*batch)
        x, y = batch
        y_hat = self.model(x)
        acc = self.acc(y_hat, y)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss = self.loss_fn(*batch)
        x, y = batch
        y_hat = self.model(x)
        acc = self.acc(y_hat, y)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        """Configure an optimizer."""
        return torch.optim.Adam(self.loss_fn.parameters(), lr=self.lr)

class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument("--size", default=1000000, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--seed", default=20200723, type=int)
    # pl.Trainer arguments.
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--devices", default="auto")
    args = parser.parse_args()

    # local_rank = int(os.environ["LOCAL_RANK"])
    # global_rank = int(os.environ["RANK"])
    
    # init_process_group(backend="gloo")

    x_train, y_train = responseGenerator(folder="./wormfunconn/atlas/", strain="unc-31").Dataset()
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.05, random_state=42)
    train_data = DataLoader(
        PsuedoDataset(x_train, torch.zeros_like(x_train), y_train),
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True, 
        num_workers = os.cpu_count()-1, 
        drop_last=True
    )

    test_data = DataLoader(
        PsuedoDataset(x_test, torch.zeros_like(x_test), y_test),
        pin_memory=True,
        shuffle=True, 
        num_workers = os.cpu_count()-1, 
        drop_last=True
    )

    model = c302.readConnectome("./data/CElegansNeuronTables.xls")
    model = c302.RecurrentNematode(model)
    summary(model)
    Guide = AutoDiagonalNormal(model)
    loss_fn = Trace_ELBO()(model, Guide)
    training_plan = PyroLightningModule(loss_fn, args.learning_rate)
    mini_batch = PsuedoDataset(x_train, torch.zeros_like(x_train), y_train)[: args.batch_size]
    loss_fn(*mini_batch)
    # Run stochastic variational inference using PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        max_epochs=args.total_epochs,
        log_every_n_steps = args.save_every, 
        callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))]
    )
    trainer.fit(training_plan, train_data)
    trainer.test(training_plan)