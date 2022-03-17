# import
from pathlib import Path
import numpy as np
import torch
import os

from S1_pnp.imgdataset import ImgDataset
from S1_pnp.networks.SpViDeCNN_network import SpViDeCNN
# process data

#train_dataset = ImgDataset('../S0_gaptv/data/train/')
#val_dataset = ImgDataset('../S0_gaptv/data/val/')



def train(train_dataset,val_dataset):
# set-up model
    hparams = {

        "train_dataset": train_dataset,
        "val_dataset": val_dataset,

        "lr": 1e-3,
        "min_epochs": 60,
        "max_epochs": 1000,
        "patience": 4,
        "batch_size": 10,
        "num_workers": 4,
        "val_sanity_checks": 1,
        "fast_dev_run": False,
        "output_path": "./S1_pnp/model-outputs/resnet",
        "log_path": "./tensorboard_logs",
        "gpu": torch.cuda.is_available(),
        # "input_layers":6,
        # "hidden_layers":128,
        # "num_blocks":4
    }


    model = SpViDeCNN(hparams=hparams)
    model.fit()
    # results
    print(f'Best IOU score is : {model.trainer_params["callbacks"][0].best_model_score}')
    # save the weights to submitssion file
    if not os.path.exists('./S1_pnp/model-outputs/resnet'):
       os.mkdir('./S1_pnp/model-outputs/resnet')
    weight_path = "./S1_pnp/model-outputs/resnet/model.pt"
    model = SpViDeCNN(hparams=hparams).load_from_checkpoint(model.trainer_params["callbacks"][0].best_model_path)
    torch.save(model.state_dict(), weight_path)


def test(dataset=False,savepath='./S1_pnp/results'):
    if not dataset:
        dataset = ImgDataset(path,mask_path,f_trans=False)
    # set-up model
    hparams = {
        "test_dataset": dataset,
        "lr": 1e-3,
        "min_epochs": 4,
        "max_epochs": 1000,
        "patience": 10,
        "batch_size": 16,
        "num_workers": 4,
        "val_sanity_checks": 0,
        "fast_dev_run": False,
        "output_path": "./S1_pnp/model-outputs",
        "log_path": "./tensorboard_logs",
        "gpu": torch.cuda.is_available(),
        # "input_layers":6,
        # "hidden_layers":16,
        # "num_blocks":4,
        "result_path":savepath
    }

    model = SpViDeCNN(hparams=hparams)
    model.load_state_dict(torch.load("/lustre/arce/X_MA/SCI_2.0_python/S1_pnp/model-outputs/resnet/model.pt"))
    model.test()


if __name__ == '__main__':
    dataset = ImgDataset('./S1_pnp/train_data')
    train_num = round(0.7*len(dataset))
    valid_num = round(0.15*len(dataset))
    print(len(dataset))
    if not len(dataset):
        Error
    train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(dataset, [train_num, valid_num,len(dataset)-train_num-valid_num], generator=torch.Generator().manual_seed(8))
    #train(train_dataset,val_dataset)

    test_dataset.dataset.test()
    test(test_dataset)
