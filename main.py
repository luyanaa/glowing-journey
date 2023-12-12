import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

from pyro.infer import SVI, Trace_ELBO, MCMC
from pyro.infer.autoguide import AutoDiagonalNormal
import pyro.optim

from utils import AtlasLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os
import c302

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
    model = c302.readConnectome("./data/CElegansNeuronTables.xls")
    dataset = AtlasLoader("./data/")


    summary(model)
    pyro.set_rng_seed(19260817)

    from pyro.infer import NUTS, autoguide
    from pyro.distributions import Empirical
    from utils import responseGenerator, PsuedoDataset

    x_train, y_train = responseGenerator(folder="./wormfunconn/atlas/", strain="unc-31").Dataset()
    dataset = PsuedoDataset(x_train, y_train)
    train_data = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False
        # sampler=DistributedSampler(dataset)
    )
    model = c302.RecurrentNematode(model)
    Guide = autoguide.AutoDiagonalNormal(model)
    optim = pyro.optim.AdagradRMSProp({})
    svi = SVI(model, Guide, optim, Trace_ELBO())
    for epoch in range(0, 1):
        print(f"Epoch {epoch} | Batchsize: {args.batch_size} | Steps: {len(train_data)}")
        # train_data.sampler.set_epoch(epoch)
        for x_train, y_train in train_data:
            print("step")
        # svi.step(x_train, y_train)
            svi.step(x_train)



#self.model = DDP(self.model, device_ids=[self.local_rank])
#destroy_process_group()
