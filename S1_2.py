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



def train(train_dataset1,val_dataset1,train_dataset2,val_dataset2):
# set-up model
    hparams = {

        "train_dataset": train_dataset1,
        "val_dataset": val_dataset1,

        "lr": 1e-3,
        "min_epochs": 4,
        "max_epochs": 1000,
        "patience": 4,
        "batch_size": 12,
        "num_workers": 4,
        "val_sanity_checks": 0,
        "fast_dev_run": False,
        "output_path": "./S1_denoiser/model-outputs",
        "log_path": "./tensorboard_logs",
        "gpu": torch.cuda.is_available(),
        "input_layers":6,
        "hidden_layers":16,
        "num_blocks":4
    }

    model = CHASTINET(hparams=hparams)
    # # run model
    # model.fit()
    # # results
    # print(f'Best IOU score is : {model.trainer_params["callbacks"][0].best_model_score}')
    # # save the weights to submitssion file
    # if not os.path.exists('./S1_denoiser/model-outputs'):
    #     os.mkdir('./S1_denoiser/model-outputs')
    # weight_path = "./S1_denoiser/model-outputs/model_1.pt"

    hparams["train_dataset"] = train_dataset2
    hparams["val_dataset"] = val_dataset2
    #hparams["lr"] = 1e-5

    # model = CHASTINET(hparams=hparams).load_from_checkpoint(model.trainer_params["callbacks"][0].best_model_path)
    # torch.save(model.state_dict(), weight_path)


    model.fit()
    # results
    print(f'Best IOU score is : {model.trainer_params["callbacks"][0].best_model_score}')
    # save the weights to submitssion file
    if not os.path.exists('./S1_denoiser/model-outputs'):
       os.mkdir('./S1_denoiser/model-outputs')
    weight_path = "./S1_denoiser/model-outputs/model_2.pt"
    model = CHASTINET(hparams=hparams).load_from_checkpoint(model.trainer_params["callbacks"][0].best_model_path)
    torch.save(model.state_dict(), weight_path)


def test(path,savepath='result',mask_path='./S0_gaptv/lesti_mask.mat', dataset=False):
    if not dataset:
        dataset = ImgDataset(path,mask_path,f_trans=False)
    # set-up model
    hparams = {
        "test_dataset": dataset,
        "lr": 1e-3,
        "min_epochs": 4,
        "max_epochs": 1000,
        "patience": 10,
        "batch_size": 8,
        "num_workers": 4,
        "val_sanity_checks": 0,
        "fast_dev_run": False,
        "output_path": "./S1_denoiser/model-outputs",
        "log_path": "./tensorboard_logs",
        "gpu": torch.cuda.is_available(),
        "input_layers":6,
        "hidden_layers":16,
        "num_blocks":4,
        "result_path":savepath
    }

    model = CHASTINET(hparams=hparams)
    model.load_state_dict(torch.load("/lustre/arce/X_MA/SCI_2.0_python/S1_denoiser/model-outputs/2022new/model_2.pt"))
    #trainer = Trainer()
    #trainer.test(model)
    model.test()


if __name__ == '__main__':
    dataset = ImgDataset('./S0_gaptv/data/trainS1_16b/', './S0_gaptv/mask_256x512.mat', '16bands')
    train_dataset2,val_dataset2,test_dataset = torch.utils.data.random_split(dataset, [587, 100, 60], generator=torch.Generator().manual_seed(8))
    train(None,None,train_dataset2,val_dataset2)
    
    test_dataset.f_trans = False
    test('S0_gaptv/data/test/','S1_denoiser/result/test_sim_newmodel',dataset=test_dataset)
    test('S0_gaptv/data/test/','S1_denoiser/result/test_sim_newmodel')
