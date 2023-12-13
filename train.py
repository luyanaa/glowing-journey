import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

import os
import pyro, pyro.optim
from pyro.infer import SMCFilter, MCMC, NUTS, SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.distributions import Empirical

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

def CacheModel(model, snapshot_path):
    if os.path.exists(snapshot_path):
        print("Loading snapshot")
        loc = f"cpu:{local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        model.load_state_dict(snapshot["MODEL_STATE"])
        epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {epochs_run}")        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
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

    if True:    
        model = c302.RecurrentNematode(model)
        Guide = AutoDiagonalNormal(model)
        optim = pyro.optim.AdagradRMSProp({})
        svi = SVI(model, Guide, optim, Trace_ELBO())
        for epoch in range(0, args.total_epochs):
            print(f"Epoch {epoch} | Batchsize: {args.batch_size} | Steps: {len(train_data)}")
            # train_data.sampler.set_epoch(epoch)
            for x_train, y_train in train_data:
                loss = svi.step(x_train, y_train)
                print("loss: %.4f" % loss / len(x_train))
            if epoch % args.save_every == 0:
                num_samples = 5000
                predictive = Predictive(model, guide=Guide, num_samples=num_samples)
                preds = predictive(x_test)


    elif False:
        nuts_kernel = NUTS(model, jit_compile=False)
        mcmc = MCMC(nuts_kernel, num_samples=50)
        mcmc.run(x_train, y_train)
        predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
        preds = predictive(x_test)


    else:
        num_particles = 50
        pyro.set_rng_seed()
        Guide = AutoDiagonalNormal(model)
        # y_train = Empirical(y_train, torch.ones(y_train.shape[:-1]), validate_args=None)
        smc = SMCFilter(model, Guide, num_particles=num_particles, max_plate_nesting=0)
        smc.init(initial=torch.zeros(300)) 
        for y in y_train[1:]:
            smc.step(y)