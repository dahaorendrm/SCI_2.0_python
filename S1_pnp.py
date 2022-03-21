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
        "output_path": "./S1_pnp/model-outputs/resnet2",
        "log_path": "./tensorboard_logs",
        "gpu": torch.cuda.is_available(),
        # "input_layers":6,
        # "hidden_layers":128,
        # "num_blocks":4
    }


    model = SpViDeCNN(hparams=hparams)
    model.cuda()
    model.fit()
    # results
    print(f'Best IOU score is : {model.trainer_params["callbacks"][0].best_model_score}')
    # save the weights to submitssion file
    if not os.path.exists('./S1_pnp/model-outputs/resnet2'):
       os.mkdir('./S1_pnp/model-outputs/resnet2')
    weight_path = "./S1_pnp/model-outputs/resnet2/model.pt"
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
        "output_path": "./S1_pnp/model-outputs2",
        "log_path": "./tensorboard_logs",
        "gpu": torch.cuda.is_available(),
        # "input_layers":6,
        # "hidden_layers":16,
        # "num_blocks":4,
        "result_path":savepath
    }
    model = SpViDeCNN(hparams=hparams)
    model.cuda()
    model.load_state_dict(torch.load("/lustre/arce/X_MA/SCI_2.0_python/S1_pnp/model-outputs/resnet2/model.pt"))
    model.test()

def compressive_model(input, mask):
        data = (
        input,
        mask #reduce loading time scio.loadmat('lesti_mask.mat')['mask']
        )
        mea = measurement.Measurement(model = 'chasti_sst', dim = 3, inputs=data, configs={'MAXV':1})
        model = recon_model.ReModel('gap','spvi')
        model.config({'lambda': 1, 'ASSESE': 1, 'ACC': False,
                'ITERs': 10, 'RECON_MODEL': 'GAP', 'RECON_DENOISER': 'spvi',
                'P_DENOISE':{'TV_WEIGHT': 0.2, 'TV_ITER': 7}})
        re = result.Result(model, mea, modul = mea.modul, orig = mea.orig)
        re = np.array(re)
        #re[re<0] = 0
        mea = np.array(mea.mea)
        # print('shape of re is '+str(mea.shape))
        return (mea,re)

def pnp_sivicnn():
    MASK = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
    COMP_FRAME = 24
    pool = multiprocessing.Pool(10)
    path = Path('../data/whispers/test')
    datalist = os.listdir(path)
    finished = []
    for idx,name in enumerate(datalist):
        if idx>5:
            break
        if name in finished:
            continue
        comp_input = []
        crops = []
        name_list = []
        imgidx = 0
        imglist = os.listdir(path/name/'HSI')
        i = 1 # There's one txt file in the folder, so we start at 1
        oneset = []
        print(f'Start process data {name}.')
        while i < len(imglist):
            img = skio.imread(path/name/'HSI'/f'{i:04d}.png')
            #print(f'1.max:{np.amax(img)}')
            img = X2Cube(img)
            #print(f'2.max:{np.amax(img)} sum {np.sum(img)}')
            if img.shape[0]!=256:
                img = skitrans.resize(img/511., (256,512))
                oneset.append(img)
            else:
                oneset.append(img/511.)
            i += 1
            if len(oneset)==COMP_FRAME:
                img = np.stack(oneset,3)
                print(f'img.shape is {img.shape}.')
                data1 = img[...,::2,:]
                #data1 = utils.selectFrames(data1)
                data2 = img[...,1::2,:]
                #data2 = utils.selectFrames(data2)
                dataset.append((data1,MASK))
                dataset.append((data2,MASK))
                oneset = []
                name_list.append(str(imgidx))
                name_list.append(str(imgidx+1))
                imgidx += 2
        print(f'Input data max is {np.amax(imgs)}.')
        print(f'{name} data finished. There are {len(crops)} sets of data now.')
        crops_mea = []
        crops_img = []
        crops_led = []

        return_crops_data = pool.starmap(compressive_model, dataset) # contain (mea, gaptv_result)
        for (mea,re) in return_crops_data:
            crops_mea.append(mea)
            crops_img.append(re)
        save_crops('S1_pnp/test_data',name_list,name,crops_mea,crops_img, crops_gt=crops)
if __name__ == '__main__':
    dataset = ImgDataset('./S1_pnp/train_data')
    train_num = round(0.7*len(dataset))
    valid_num = round(0.15*len(dataset))
    print(len(dataset))
    if not len(dataset):
        Error
    train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(dataset, [train_num, valid_num,len(dataset)-train_num-valid_num], generator=torch.Generator().manual_seed(8))
    train(train_dataset,val_dataset)

    test_dataset.dataset.test()
    test(test_dataset)
