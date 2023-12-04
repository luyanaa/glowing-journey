import pyro
import pyro.distributions as dist
from pyro.infer import SMCFilter, MCMC, NUTS
import torch


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.loss = F.mse_loss
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])
        if svi:
            self.svi=SVI(self.model, AutoDiagonalNormal(self.model), optimizer, loss=Trace_ELBO())
        else:
            self.svi=MCMC

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


#    def _run_epoch(self, epoch):

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")



# Particle Filter can be realized with Pyro Sequential MC
def ParticleFilter(model, num_particles=114, seed=19260817):
    pyro.set_rng_seed(seed)

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
                    model(state["z"]), torch.tensor([1.0, 1.0])
                ).to_event(self.event_dim),
            )

    guide = Guide(model)
    sample_distribution = dist.Empirical(samples, log_weights, validate_args=None)

    smc = SMCFilter(model, guide, num_particles=num_particles, max_plate_nesting=0)
    return smc

def SMCtrain(self, smc):
    smc.init(initial=)
    for y in ys[1:]:
        smc.step(y)

    z = smc.get_empirical()["z"]

def MCMC(model):
    nuts_kernel = NUTS(model, jit_compile=False)
    mcmc = MCMC(nuts_kernel, num_samples=50)
    mcmc.run(x_train, y_train)

# SVI step, supporting batch
def _run_epoch_SVI(self, epoch, svi):
    b_sz = len(next(iter(self.train_data))[0])
    print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
    self.train_data.sampler.set_epoch(epoch)
    for x_train, y_train in self.train_data:
        x_train = x_train.to(self.local_rank)
        y_train = y_train.to(self.local_rank)
        sample_distribution = dist.Empirical(samples, log_weights, validate_args=None)

        svi.step(x_train, y_train)

def train_SVI(self, max_epochs: int):
    for epoch in range(self.epochs_run, max_epochs):
        _run_epoch_SVI(epoch)
        if self.local_rank == 0 and epoch % self.save_every == 0:
            self._save_snapshot(epoch)