# import
from pathlib import Path
import numpy as np
#from . import utils
import torch
import os

from S1_denoiser.imgdataset import ImgDataset
from S1_denoiser.networks.chasti_network import CHASTINET
# process data

def test(path,savepath='result',mask_path='../S0_gaptv/lesti_mask.mat'):
    dataset = ImgDataset(path,mask_path,f_trans=False)
    # set-up model
    hparams = {
        "test_dataset": dataset,
        "lr": 1e-3,
        "min_epochs": 4,
        "max_epochs": 1000,
        "patience": 10,
        "batch_size": 4,
        "num_workers": 2,
        "val_sanity_checks": 0,
        "fast_dev_run": False,
        "output_path": "model-outputs",
        "log_path": "tensorboard_logs",
        "gpu": torch.cuda.is_available(),
        "input_layers":4,
        "hidden_layers":32,
        "num_blocks":4,
        "result_path":savepath
    }

    model = CHASTINET(hparams=hparams)
    model.load_state_dict(torch.load("/lustre/arce/X_MA/SCI_2.0_python/S1_denoiser/model-outputs/model_2.pt"))
    #trainer = Trainer()
    #trainer.test(model)
    model.test()

if __name__ == '__main__':
    test('../S0_gaptv/data/test/')
