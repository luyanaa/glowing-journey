import argparse
import os

import pyro
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateFinder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torchinfo import summary

import c302
from pyro_svi_c302 import DEFAULT_DATA_PATH, _TupleDataset
from soft_dtw_cuda import SoftDTW
from utils import AtlasLoader, responseGenerator


def _save_snapshot(snapshot_path, model, epoch):
    target = model.module if hasattr(model, "module") else model
    snapshot = {
        "MODEL_STATE": target.state_dict(),
        "EPOCHS_RUN": epoch,
    }
    torch.save(snapshot, snapshot_path)
    print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")


class PyroLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper driving Pyro ELBO optimization on GCaMP trials."""

    def __init__(self, loss_fn: pyro.infer.elbo.ELBOModule, lr: float):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = loss_fn.model
        self.guide = loss_fn.guide
        self.lr = lr
        self.predictive = pyro.infer.Predictive(
            self.model, guide=self.guide, num_samples=1, return_sites=("_RETURN",)
        )
        self.acc = SoftDTW(torch.cuda.is_available())

    def forward(self, stim, noise_scale=None):
        return self.predictive(ExternalInput=stim, mask=None, y=None, noise_scale=noise_scale)["_RETURN"].squeeze(0)

    def _shared_step(self, batch, stage: str):
        stim, y, mask, noise = batch
        loss = self.loss_fn(ExternalInput=stim, mask=mask, y=y, noise_scale=noise)
        with torch.no_grad():
            y_hat = self.model(ExternalInput=stim, mask=None, y=None, noise_scale=noise)
            masked_hat = y_hat * mask.float()
            masked_y = y * mask.float()
            acc = self.acc(masked_hat, masked_y).mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_dtw", acc, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
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
    parser = argparse.ArgumentParser(description="Lightning Pyro training job")
    parser.add_argument("total_epochs", nargs="?", type=int, default=10)
    parser.add_argument("save_every", nargs="?", type=int, default=5)
    parser.add_argument("--batch_size", "--batch-size", default=16, type=int)
    parser.add_argument("--source", choices=["randi", "synthetic"], default="randi")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, type=str)
    parser.add_argument("--strain", default="wild-type", type=str)
    parser.add_argument("--norm-mode", choices=["dff_safe", "dff_strict", "robust_z"], default="dff_safe")
    parser.add_argument("--max-recordings", default=None, type=int)
    parser.add_argument("--learning_rate", default=5e-3, type=float)
    parser.add_argument("--substeps", default=4, type=int)
    parser.add_argument("--dt-obs", default=0.5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--devices", default="auto")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    pyro.clear_param_store()

    if args.source == "randi":
        dataset = AtlasLoader(
            path=args.data_path,
            strain=args.strain,
            norm_mode=args.norm_mode,
            max_recordings=args.max_recordings,
        ).build_trial_dataset()
        idx = list(range(len(dataset)))
        train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=args.seed)
        train_ds = Subset(dataset, train_idx)
        test_ds = Subset(dataset, test_idx)
    else:
        x_all, y_all, m_all = responseGenerator(
            folder="./wormfunconn/atlas/", strain=args.strain, size=100, dt=args.dt_obs
        ).Dataset(return_mask=True)
        noise_all = torch.full((x_all.shape[0], x_all.shape[2]), 0.05, dtype=torch.float32)
        idx = list(range(len(x_all)))
        train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=args.seed)
        train_ds = _TupleDataset(x_all[train_idx], y_all[train_idx], m_all[train_idx], noise_all[train_idx])
        test_ds = _TupleDataset(x_all[test_idx], y_all[test_idx], m_all[test_idx], noise_all[test_idx])

    pin = torch.cuda.is_available()
    train_data = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        pin_memory=pin,
        shuffle=True,
        drop_last=len(train_ds) >= args.batch_size,
    )
    test_data = DataLoader(
        test_ds,
        batch_size=min(args.batch_size, len(test_ds)),
        pin_memory=pin,
        shuffle=False,
    )

    step_model = c302.readConnectome("./data/CElegansNeuronTables.xls")
    model = c302.RecurrentNematode(step_model, dt_obs=args.dt_obs, substeps=args.substeps)
    summary(model)
    guide = AutoDiagonalNormal(model, init_scale=0.02)
    loss_fn = Trace_ELBO()(model, guide)
    training_plan = PyroLightningModule(loss_fn, args.learning_rate)

    # Materialize Pyro guide parameters before Trainer starts
    stim_b, y_b, mask_b, noise_b = next(iter(train_data))
    loss_fn(ExternalInput=stim_b, mask=mask_b, y=y_b, noise_scale=noise_b)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        max_epochs=args.total_epochs,
        log_every_n_steps=max(1, args.save_every),
    )
    trainer.fit(training_plan, train_dataloaders=train_data, val_dataloaders=test_data)
    trainer.test(training_plan, dataloaders=test_data)
