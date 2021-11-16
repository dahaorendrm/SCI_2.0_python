# import
from pathlib import Path
import numpy as np
import torch
import os

from imgdataset import ImgDataset
from networks.chasti_network import CHASTINET
import loss
# process data

train_dataset = ImgDataset('../S0_gaptv/data/train/')
val_dataset = ImgDataset('../S0_gaptv/data/val/')

# set-up model
hparams = {

    "train_dataset": train_dataset,
    "val_dataset": val_dataset,

    "lr": 1e-3,
    "min_epochs": 4,
    "max_epochs": 1000,
    "patience": 3,
    "batch_size": 8,
    "num_workers": 16,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": "model-outputs",
    "log_path": "tensorboard_logs",
    "gpu": torch.cuda.is_available(),
    "input_layers":3,
    "hidden_layers":64,
    "num_blocks":4
}

model = CHASTINET(hparams=hparams)

# run model
model.fit()
# results
print(f'Best IOU score is : {model.trainer_params["callbacks"][0].best_model_score}')
# save the weights to submitssion file
if not os.path.exists('model-outputs'):
    os.mkdir('model-outputs')
weight_path = "model-outputs/model.pt"
model = CHASTINET(hparams=hparams).load_from_checkpoint(model.trainer_params["callbacks"][0].best_model_path)
torch.save(model.state_dict(), weight_path)
