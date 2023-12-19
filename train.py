import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import os
import time
import pyro, pyro.optim
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive, SMCFilter
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.distributions import Empirical
import pyro.distributions as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from torchinfo import summary

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from utils import AtlasLoader, responseGenerator, PsuedoDataset
import c302

# Stochastic Variational Inference, Monte Carlo Markov Chain and Sequential Monte Carlo Filter.
epochs_run = 0

def _save_snapshot(snapshot_path, model, epoch):
    snapshot = {
            "MODEL_STATE": model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
    torch.save(snapshot, snapshot_path)
    print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")   

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    # local_rank = int(os.environ["LOCAL_RANK"])
    # global_rank = int(os.environ["RANK"])
    
    # init_process_group(backend="gloo")

    x_train, y_train = responseGenerator(folder="./wormfunconn/atlas/", strain="unc-31").Dataset()
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.05, random_state=42)
    dataset = PsuedoDataset(x_train, y_train)
    train_data = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False
        # sampler=DistributedSampler(dataset)
    )

    model = c302.readConnectome("./data/CElegansNeuronTables.xls")
    summary(model)

    if False:    
        model = c302.RecurrentNematode(model)
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")
            model = model.to("cuda")
        Guide = AutoDiagonalNormal(model)
        optim = pyro.optim.AdagradRMSProp({})
        svi = SVI(model, Guide, optim, Trace_ELBO())
        for epoch in range(0, args.total_epochs):
            print(f"Epoch {epoch} | Batchsize: {args.batch_size} | Steps: {len(train_data)}")
            # train_data.sampler.set_epoch(epoch)
            for x_train, y_train in train_data:
                VoltageClamp, ExternalInput = x_train, torch.zeros_like(x_train)
                if torch.cuda.is_available():
                    VoltageClamp, ExternalInput, y_train = VoltageClamp.cuda(), ExternalInput.cuda(), y_train.cuda() 
                loss = svi.step(VoltageClamp, ExternalInput, y=y_train)
                print("loss: %.4f" % loss )

    else:
        num_particles = 300
        pyro.set_rng_seed(time.time())
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")
            model = model.to("cuda")

        class Guide:
            def init(self, state, initial):
                self.t = 0
                pyro.sample("z_init", dist.Delta(initial, event_dim=1))

            def step(self, state, time, Prev, ExternalInput=None, VoltageClamp=None, y=None, mask=None ):
                self.t = time
                print(self.t)
                sigma = pyro.sample("sigma_%d" % self.t, dist.Uniform(70.71, 223.61))
                # Proposal distribution
                pyro.sample(
                    "z_{}".format(self.t),
                    dist.Normal(
                        y,  1 / (sigma * sigma)
                    ).to_event(2),
                )
        guide = Guide
        state = {}


        smc = SMCFilter(model, guide, num_particles=num_particles, max_plate_nesting=0)
        model.init(state=state, initial=torch.zeros(300))
        smc.init(state=state, initial=torch.zeros(300)) 
        print(x_train[:, 1-1, :].shape)
        which = 1
        for index in range(1, x_train.shape[1]):sm
            smc.step(time=index, Prev=x_train[:, index-1, :].cuda(), ExternalInput=torch.zeros_like(x_train[:, index, :]).cuda(), VoltageClamp=x_train[:, index, :].cuda(), y=y_train[:, index, :].cuda())