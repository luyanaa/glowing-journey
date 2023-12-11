import pyro
import pyro.distributions as dist
from pyro.infer import SMCFilter, MCMC, NUTS, SVI, Trace_ELBO
import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import torch.functional as F

# Stochastic Variational Inference, Monte Carlo Markov Chain and Sequential Monte Carlo Filter.
epochs_run = 0

def _save_snapshot(snapshot_path, model, epoch):
    snapshot = {
            "MODEL_STATE": model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
    torch.save(snapshot, snapshot_path)
    print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")


class Guide:
    def __init__(self, model):
            self.model = model

    def init(self, state, initial):
        self.t = 0
        pyro.sample("z_init", dist.Delta(initial, self.event_dim))

    def step(self, state, y=None):
        self.t += 1
            # Proposal distribution
        pyro.sample(
                "z_{}".format(self.t),
                dist.Normal(
                    self.model(state["z"]), torch.tensor([1.0, 1.0])
                ).to_event(self.event_dim),
            )

# Particle Filter can be realized with Pyro Sequential MC
def ParticleFilter(model, y_train, num_particles=114, seed=19260817):
    pyro.set_rng_seed(seed)


    guide = Guide(model)
    y_train = dist.Empirical(y_train, torch.ones(y_train.shape[:-1]), validate_args=None)

    smc = SMCFilter(model, guide, num_particles=num_particles, max_plate_nesting=0)
    smc.init(initial=torch.zeros(302)) 

    for y in y_train[1:]:
        smc.step(y)

def SVIRunner(model, max_epochs):
    svi = SVI(model, Guide, pyro.optim.RMSprop(), Trace_ELBO())
    for epoch in range(epochs_run, max_epochs):
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        train_data.sampler.set_epoch(epoch)
        for x_train in train_data:
            x_train = x_train.to(local_rank)
            svi.step(x_train)

        if local_rank == 0 and epoch % save_every == 0:
            _save_snapshot(epoch)

# Using Monte Carlo Markov Chain
def train_MCMC(model, x_train, y_train):
    nuts_kernel = NUTS(model, jit_compile=False)
    mcmc = MCMC(nuts_kernel, num_samples=50)
    mcmc.run(x_train, y_train)

    return 

def CacheModel(model, snapshot_path):
    if os.path.exists(snapshot_path):
        print("Loading snapshot")
        loc = f"cpu:{local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        model.load_state_dict(snapshot["MODEL_STATE"])
        epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {epochs_run}")        