# import
from pathlib import Path
import numpy as np
import utils
import torch
import os

from ImgDataset import ImgDataset
from networks.SpecConvModel import SpecConvModel
import loss
# process data

#dataset = ImgDataset('./data/train/feature', './data/train/label')
dataset = ImgDataset('../S0_gaptv/data/trainS3/img_n', '../S0_gaptv/data/trainS3/gt')
train_dataset,val_dataset = torch.utils.data.random_split(dataset, [370, 90], generator=torch.Generator().manual_seed(8))

# set-up model
hparams = {
    # Required hparams
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    # Optional hparams
    "backbone": "resnet34",
    "weights": "imagenet",
    "lr": 1e-3,
    "min_epochs": 4,
    "max_epochs": 1000,
    "patience": 10,
    "batch_size": 32,
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

# run model
model.fit()
# results
print(f'Best IOU score is : {model.trainer_params["callbacks"][0].best_model_score}')
# save the weights to submitssion file
if not os.path.exists('model-outputs'):
    os.mkdir('model-outputs')
weight_path = "model-outputs/model.pt"
model = SpecConvModel(hparams=hparams).load_from_checkpoint(model.trainer_params["callbacks"][0].best_model_path)
torch.save(model.state_dict(), weight_path)
