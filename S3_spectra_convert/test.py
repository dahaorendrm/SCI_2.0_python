# import
from pathlib import Path
import numpy as np
import utils
import torch
import os

from ImgDataset import TestDataset
from networks.SpecConvModel import SpecConvModel
import loss
import pytorch_lightning as pl
# process data


test_dataset = TestDataset('./data/test/feature','./data/test/label')


# set-up model
hparams = {
    # Required hparams
    "test_dataset": test_dataset,
    # Optional hparams
    "backbone": "resnet34",
    "weights": "imagenet",
    "lr": 1e-3,
    "min_epochs": 4,
    "max_epochs": 1000,
    "patience": 10,
    "batch_size": 1,
    "num_workers": 16,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": "model-outputs",
    "log_path": "tensorboard_logs",
    "gpu": torch.cuda.is_available(),
    "in_channels":8,
    "out_channels":25
}

model = SpecConvModel(hparams=hparams)
model.load_state_dict(torch.load("model-outputs/model.pt"))

trainer = pl.Trainer()
trainer.test(model)
