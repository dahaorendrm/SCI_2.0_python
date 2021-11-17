# import
from pathlib import Path
import numpy as np
import utils
import torch
import os

from imgdataset import ImgDataset
from networks.chasti_network import CHASTINET
import loss
# process data

dataset = ImgDataset('../S0_gaptv/data/test/')

# set-up model
hparams = {

    "test_dataset": dataset,

    "lr": 1e-3,
    "min_epochs": 4,
    "max_epochs": 1000,
    "patience": 10,
    "batch_size": 1,
    "num_workers": 2,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": "model-outputs",
    "log_path": "tensorboard_logs",
    "gpu": torch.cuda.is_available(),
    "input_layers":4,
    "hidden_layers":64,
    "num_blocks":4
}




model = CHASTINET(hparams=hparams)
model.load_state_dict(torch.load("model-outputs/S1_noise/model.pt"))

#trainer = Trainer()
#trainer.test(model)
model.test()
