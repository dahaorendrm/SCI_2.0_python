# import
from pathlib import Path
import numpy as np
from . import utils
import torch
import os

from .imgdataset import ImgDataset
from .networks.chasti_network import CHASTINET
from . import loss
# process data

def test(path):
    dataset = ImgDataset(path)
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
    model.load_state_dict(torch.load("/lustre/arce/X_MA/SCI_2.0_python/S1_denoiser/model-outputs/model.pt"))
    #trainer = Trainer()
    #trainer.test(model)
    model.test()

if __name__ == '__main__':
    test('../S0_gaptv/data/test/')
