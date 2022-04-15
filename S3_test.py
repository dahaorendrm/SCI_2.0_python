# import
from pathlib import Path
import numpy as np
import torch
import os
import pytorch_lightning as pl

from S3_spectra_convert.ImgDataset import ImgDataset,TestDataset
from S3_spectra_convert.networks.SpecConvModel import SpecConvModel
from S3_spectra_convert import loss,utils
# process data

def test(datapath='./S2_flow_predict/result/re',refpath='./S0_gaptv/data/test/gt',savepath='S1_denoiser/result/re'):
    # test_dataset = TestDataset('./data/test/feature','./data/test/label')
    test_dataset = ImgDataset(datapath, refpath, f_trans = False)
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
        "output_path": "./S3_spectra_convert/model-outputs",
        "log_path": "tensorboard_logs",
        "gpu": torch.cuda.is_available(),
        "in_channels":8,
        "out_channels":25,
        'savepath':savepath
    }
    model = SpecConvModel(hparams=hparams)
    model.load_state_dict(torch.load("/lustre/arce/X_MA/SCI_2.0_python/S3_spectra_convert/model-outputs/gaptv_train_ssim/model.pt"))
    #trainer = Trainer()
    #trainer.test(model)
    model.test()

def test_paper(savepath='S1_denoiser/result/re_paper'):
    # test_dataset = TestDataset('./data/test/feature','./data/test/label')
    test_dataset = PaperDataset('/lustre/arce/X_MA/SCI_2.0_python/S3_spectra_convert/data_paperpnp/3D_Doll_center.mat')
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
        "output_path": "./S3_spectra_convert/model-outputs",
        "log_path": "tensorboard_logs",
        "gpu": torch.cuda.is_available(),
        "in_channels":8,
        "out_channels":25,
        'savepath':savepath
    }
    model = SpecConvModel(hparams=hparams)
    model.load_state_dict(torch.load("/lustre/arce/X_MA/SCI_2.0_python/S3_spectra_convert/model-outputs/gaptv_train_ssim/model.pt"))
    #trainer = Trainer()
    #trainer.test(model)
    model.test()

if __name__=='__main__':
    #test('./S2_flow_predict/result/re','./S1_pnp/data/test/gt','S3_spectra_convert/result/re')
    test_paper()
