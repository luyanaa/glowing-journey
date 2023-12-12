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

# Particle Filter can be realized with Pyro Sequential MC
def ParticleFilter(model, y_train, num_particles=114, seed=19260817):
    pyro.set_rng_seed(seed)


    guide = Guide(model)
    y_train = dist.Empirical(y_train, torch.ones(y_train.shape[:-1]), validate_args=None)

    smc = SMCFilter(model, guide, num_particles=num_particles, max_plate_nesting=0)
    smc.init(initial=torch.zeros(302)) 

    for y in y_train[1:]:
        smc.step(y)

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