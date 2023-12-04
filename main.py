import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import train

from pyro.infer import SVI, Trace_ELBO, MCMC
from pyro.infer.autoguide import AutoDiagonalNormal
import pyro.optim

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import c302

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    init_process_group(backend="gloo")
    model = c302.readConnectome("./data/CElegansNeuronTables.xls")
    dataset = utils.load_randi()
    optimizer = pyro.optim.RMSprop()
    train_data = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

    trainer = Trainer(model, train_data, optimizer, args.save_every, args.snapshot_path)
    trainer.train(args.total_epochs)
    destroy_process_group()
