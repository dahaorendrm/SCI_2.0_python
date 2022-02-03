# import
from pathlib import Path
import numpy as np
import torch
import os

from S3_spectra_convert.ImgDataset import ImgDataset
from S3_spectra_convert.networks.SpecConvModel import SpecConvModel
import S3_spectra_convert.loss
from S3_spectra_convert import utils
# process data

#dataset = ImgDataset('./data/train/feature', './data/train/label')
dataset = ImgDataset('./S0_gaptv/data/trainS3/img_n', './S0_gaptv/data/trainS3/gt')
train_dataset,val_dataset = torch.utils.data.random_split(dataset, [370, 90], generator=torch.Generator().manual_seed(8))

# set-up model
hparams = {
    # Required hparams
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    # Optional hparams
    "backbone": "resnext50_32x4d",
    "weights": "imagenet",
    "lr": 1e-3,
    "min_epochs": 4,
    "max_epochs": 1000,
    "patience": 15,
    "batch_size": 32,
    "num_workers": 4,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": "./S3_spectra_convert/model-outputs",
    "log_path": "./S3_spectra_convert/tensorboard_logs",
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
if not os.path.exists('./S3_spectra_convert/model-outputs'):
    os.mkdir('./S3_spectra_convert/model-outputs')
weight_path = "./S3_spectra_convert/model-outputs/model.pt"
model = SpecConvModel(hparams=hparams).load_from_checkpoint(model.trainer_params["callbacks"][0].best_model_path)
torch.save(model.state_dict(), weight_path)
