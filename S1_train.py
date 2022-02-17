# import
from pathlib import Path
import numpy as np
import torch
import os

from S1_denoiser.imgdataset import ImgDataset
from S1_denoiser.networks.chasti_network import CHASTINET
# process data

#train_dataset = ImgDataset('../S0_gaptv/data/train/')
#val_dataset = ImgDataset('../S0_gaptv/data/val/')
dataset = ImgDataset('./S0_gaptv/data/trainS1_rgb/')
train_dataset1,val_dataset1 = torch.utils.data.random_split(dataset, [2635, 650], generator=torch.Generator().manual_seed(8))

dataset = ImgDataset('./S0_gaptv/data/trainS1_16b/', './S0_gaptv/mask_256x512.mat', '16bands')
train_dataset2,val_dataset2 = torch.utils.data.random_split(dataset, [307, 76], generator=torch.Generator().manual_seed(8))

dataset = ImgDataset('./S0_gaptv/data/trainS1_16b_2/', './S0_gaptv/mask_256x512.mat', '16bands')
train_dataset3,val_dataset3 = torch.utils.data.random_split(dataset, [344, 86], generator=torch.Generator().manual_seed(8))

train_dataset2 = torch.utils.data.ConcatDataset([train_dataset2, train_dataset3])
val_dataset2   = torch.utils.data.ConcatDataset([val_dataset2, val_dataset3])
# set-up model


hparams = {

    "train_dataset": train_dataset1,
    "val_dataset": val_dataset1,

    "lr": 1e-3,
    "min_epochs": 4,
    "max_epochs": 1000,
    "patience": 6,
    "batch_size": 8,
    "num_workers": 2,
    "val_sanity_checks": 0,
    "fast_dev_run": False,
    "output_path": "./S1_denoiser/model-outputs",
    "log_path": "./tensorboard_logs",
    "gpu": torch.cuda.is_available(),
    "input_layers":4,
    "hidden_layers":32,
    "num_blocks":4
}

model = CHASTINET(hparams=hparams)
# run model
model.fit()
# results
print(f'Best IOU score is : {model.trainer_params["callbacks"][0].best_model_score}')
# save the weights to submitssion file
if not os.path.exists('./S1_denoiser/model-outputs'):
    os.mkdir('./S1_denoiser/model-outputs')
weight_path = "./S1_denoisermodel-outputs/model_1.pt"

hparams["train_dataset"] = train_dataset2
hparams["val_dataset"] = val_dataset2
hparams["lr"] = 1e-5

model = CHASTINET(hparams=hparams).load_from_checkpoint(model.trainer_params["callbacks"][0].best_model_path)
torch.save(model.state_dict(), weight_path)


model.fit()
# results
print(f'Best IOU score is : {model.trainer_params["callbacks"][0].best_model_score}')
# save the weights to submitssion file
if not os.path.exists('./S1_denoiser/model-outputs'):
   os.mkdir('./S1_denoiser/model-outputs')
weight_path = "./S1_denoiser/model-outputs/model_2.pt"
model = CHASTINET(hparams=hparams).load_from_checkpoint(model.trainer_params["callbacks"][0].best_model_path)
torch.save(model.state_dict(), weight_path)