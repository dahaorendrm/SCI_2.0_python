# import
from pathlib import Path
import numpy as np
import utils
import torch
import os

from ImgDataset import ImgDataset,TestDataset
from networks.SpecConvModel import SpecConvModel
import loss
import pytorch_lightning as pl
# process data


# test_dataset = TestDataset('./data/test/feature','./data/test/label')
test_dataset = ImgDataset('../S2_flow_predict/result/re_spct','../S0_gaptv/data/test/gt', f_trans = False)

# set-up model
hparams = {
    # Required hparams
    "test_dataset": test_dataset,
    # Optional hparams
    # "backbone": "resnet34",
    "backbone": "resnext50_32x4d",
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

model.load_state_dict(torch.load("model-outputs/gaptv_train_ssim/model.pt"))
#trainer = Trainer()
#trainer.test(model)
model.test()
