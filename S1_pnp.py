# import
from pathlib import Path
import numpy as np
import torch
import os
import scipy.io as scio
import multiprocessing,threading,queue
from skimage import io as skio
from skimage import transform as skitrans

from S0_gaptv.func import utils,recon_model,result,measurement
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

def X2Cube(img,B=[4, 4],skip = [4, 4],bandNumber=16):
    '''
    This function came with the whispers datasets
    '''
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//4, N//4,bandNumber )
    return DataCube

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

def save_crops(path, name, idx, gt=None, mea, re):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            pass
    name = '_'.join((name,'%.4d'%(idx)+'.tiff'))
    os.mkdir(path+'/mea/') if not os.path.exists(path+'/mea') else None
        tifffile.imwrite(path+'/mea/'+name,mea)
    os.mkdir(path+'/img_n/') if not os.path.exists(path+'/img_n') else None
        tifffile.imwrite(path+'/img_n/'+name,re)
    if gt:
        os.mkdir(path+'/gt/') if not os.path.exists(path+'/gt') else None
        tifffile.imwrite(path+'/gt/'+name,gt)

def pnp_sivicnn():
    MASK = scio.loadmat('/lustre/arce/X_MA/SCI_2.0_python/S0_gaptv/lesti_mask.mat')['mask']
    COMP_FRAME = 24
    pool = multiprocessing.Pool(10)
    path = Path('../data/whispers/test')
    datalist = os.listdir(path)
    finished = []
    for idx,name in enumerate(datalist):
        if idx>2:
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
        dataset = []
        crops = []
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
                ranp = np.random.uniform(0,256,4)
                print(f'img.shape is {img.shape}.')
                for po in ranp:
                    data1 = img[:,po:po+256,::2,:]
                    #data1 = utils.selectFrames(data1)
                    data2 = img[:,po:po+256,1::2,:]
                    #data2 = utils.selectFrames(data2)
                    dataset.append((data1,MASK))
                    dataset.append((data2,MASK))
                    crops.append(data1)
                    crops.append(data2)
                oneset = []

        print(f'Input data max is {np.amax(img)}.')
        for idx,data_1 in enumerate(dataset):
            (mea,re) = compressive_model(*data_1)
            save_crops('S1_pnp/test_data', name, idx, data_1[0], mea, re)

if __name__ == '__main__':
    #dataset = ImgDataset('./S1_pnp/train_data')
    #train_num = round(0.7*len(dataset))
    #valid_num = round(0.15*len(dataset))
    #print(len(dataset))
    #if not len(dataset):
    #    Error
    #train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(dataset, [train_num, valid_num,len(dataset)-train_num-valid_num], generator=torch.Generator().manual_seed(8))
    #train(train_dataset,val_dataset)

    #test_dataset.dataset.test()
    #test(test_dataset)
    pnp_sivicnn()
