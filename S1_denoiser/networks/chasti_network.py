import torch
import numpy as np
from pathlib import Path

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from . import Resblock
from .loss import *
from utils import *

class CHASTINET(pl.LightningModule):
    def __init__(self, hparams):
        super(CHASTINET, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.learning_rate = self.hparams.get("lr", 1e-3)
        self.max_epochs = self.hparams.get("max_epochs", 1000)
        self.min_epochs = self.hparams.get("min_epochs", 6)
        self.patience = self.hparams.get("patience", 4)
        self.num_workers = self.hparams.get("num_workers", 8)
        self.batch_size = self.hparams.get("batch_size", 32)
        self.train_dataset = self.hparams.get("train_dataset",None)
        self.val_dataset = self.hparams.get("val_dataset",None)
        self.test_dataset = self.hparams.get("test_dataset",None)
        self.output_path = self.hparams.get("output_path", "model-outputs")
        self.gpu = self.hparams.get("gpu", False)
        self.input_layers = self.hparams.get("input_layers", 4)
        self.hidden_layers = self.hparams.get("hidden_layers", 64)
        self.num_blocks = self.hparams.get("num_blocks", 4)
        self.resultpath = Path(self.hparams.get("result_path", 'result/re'))

        # Where final model will be saved
        self.output_path = Path.cwd() / self.output_path
        self.output_path.mkdir(exist_ok=True)

        # Track validation IOU globally (reset each epoch)
        self.psnr_val = []
        self.ssim_val = []

        # Instantiate datasets, model, and trainer params
        self.model = self._prepare_model()
        self.trainer_params = self._get_trainer_params()

    ## Required LightningModule methods ##

    def forward(self, image):
        # Forward pass
        return self.model(image)

    def selectFrames(self, gt):
        '''
        create reference label from gt to only keep desired frames
        '''
        new_size = list(gt.size())
        new_size.pop(3)
        new_gt = torch.empty(new_size, dtype=torch.float, device = gt.get_device())
        ch_idx = 0
        for f_idx in range(gt.size()[4]):
            new_gt[:,:,:,f_idx] = gt[:,:,:,ch_idx,f_idx]
            ch_idx+=1
            if ch_idx==gt.size()[3]:
                ch_idx = 0
        return new_gt


    def training_step(self, batch, batch_idx):
        self.model.train()
        torch.set_grad_enabled(True)
        y = batch['label'].float()
        y = self.selectFrames(y)
        mea = batch['mea'].float()
        if self.gpu:
            y = y.cuda(non_blocking=True)
        preds = []
        loss_li = []
        criterion = XEDiceLoss()
        for i in range(batch['img_n'].shape[3]):
            img_pre = batch['img_n'][...,i-1].float() if i>0 else batch['img_n'][...,i].float()
            img_cur = batch['img_n'][...,i].float()
            img_lat = batch['img_n'][...,i+1].float() if i<batch['img_n'].shape[3]-1 else batch['img_n'][...,i].float()
            oth_n = batch['oth_n'][...,i].float()
            mask = batch['mask'][...,i].float()
            input = torch.stack((img_cur, img_pre, img_lat, mea,mask,oth_n),1)
            if self.gpu:
                input= input.cuda(non_blocking=True)
            pred = self.model(input)
            loss_li.append(criterion(pred, y[...,i].unsqueeze(1)))
            #preds.append(torch.squeeze(pred,1))
        #preds = torch.stack(preds,3)

        # print(f'shape of preds is {preds.size()}, shape of y is {y.size()}')
        loss = sum(loss_li) / len(loss_li)
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss


    def validation_step(self, batch, batch_idx):
        self.model.eval()
        torch.set_grad_enabled(False)
        y = batch['label'].float()
        y = self.selectFrames(y)
        mea = batch['mea'].float()
        if self.gpu:
            y = y.cuda(non_blocking=True)
        preds = []
        for i in range(batch['img_n'].shape[3]):
            img_pre = batch['img_n'][...,i-1].float() if i>0 else batch['img_n'][...,i].float()
            img_cur = batch['img_n'][...,i].float()
            img_lat = batch['img_n'][...,i+1].float() if i<batch['img_n'].shape[3]-1 else batch['img_n'][...,i].float()
            oth_n = batch['oth_n'][...,i].float()
            mask = batch['mask'][...,i].float()
            input = torch.stack((img_cur, img_pre, img_lat, mea,mask,oth_n),1)
            if self.gpu:
                input= input.cuda(non_blocking=True)
            pred = self.model(input)
            preds.append(torch.squeeze(pred,1))
        preds = torch.stack(preds,3)
        # saveintemp(preds.cpu().numpy(),batch['id'][0])
        #print(f'shape of preds is {preds.size()}, label is {y.size()}')
        psnr_val_n = calculate_psnr(batch['img_n'].cpu().numpy(), y.cpu().numpy())
        preds = preds.cpu().numpy()
        y = y.cpu().numpy()
        preds = np.squeeze(np.moveaxis(preds,0,-1))
        y = np.squeeze(np.moveaxis(y,0,-1))
        #print(preds.shape)
        #print(y.shape)
        psnr_val = calculate_psnr(preds, y)
        ssim_val = calculate_ssim(preds, y)
        self.psnr_val.append(psnr_val)
        self.ssim_val.append(ssim_val)
        self.log(
            "val_psnr_in",
            psnr_val_n,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_psnr_step",
            psnr_val,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_ssim_step",
            ssim_val,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        return (preds,psnr_val)

    def test_step(self, batch, batch_idx):
        save_name = batch['id'][0]
        self.model.eval()
        torch.set_grad_enabled(False)
        if not batch['label'] is False:
            y = batch['label'].float()
            y = self.selectFrames(y)
        else:
            y=False
        mea = batch['mea'].float()
        if self.gpu:
            y = y.cuda(non_blocking=True) if not y is False else False
        preds = []
        for i in range(batch['img_n'].shape[3]):
            img_pre = batch['img_n'][...,i-1].float() if i>0 else batch['img_n'][...,i].float()
            img_cur = batch['img_n'][...,i].float()
            img_lat = batch['img_n'][...,i+1].float() if i<batch['img_n'].shape[3]-1 else batch['img_n'][...,i].float()
            oth_n = batch['oth_n'][...,i].float()
            mask = batch['mask'][...,i].float()
            input = torch.stack((img_cur, img_pre, img_lat, mea,mask,oth_n),1)
            if self.gpu:
                input= input.cuda(non_blocking=True)
            pred = self.model(input)
            preds.append(torch.squeeze(pred,1))
        preds = torch.stack(preds,3)
        #saveintemp(preds.cpu().numpy(),batch['id'][0])
        #saveintemp(batch['img_n'].cpu().numpy(),'img_n_'+batch['id'][0])
        preds = preds.cpu().numpy()
        preds = np.squeeze(np.moveaxis(preds,0,-1)) # *0.71
        #tifffile.imwrite(self.resultpath/f"{batch['id'][0]}.tiff",preds)
        psnr_val = None
        if not y is False:
            #saveintemp(y.cpu().numpy(),'ref_'+batch['id'][0])
            ref_y = y.cpu().numpy()
            ref_y = np.squeeze(np.moveaxis(ref_y,0,-1))
            img_n = batch['img_n'].cpu().numpy()
            img_n = np.squeeze(np.moveaxis(img_n,0,-1))
            psnr_in = calculate_psnr(img_n, ref_y)
            ssim_in = calculate_ssim(img_n, ref_y)
            psnr_re = calculate_psnr(preds, ref_y)
            ssim_re = calculate_ssim(preds, ref_y)

            psnr_n,ssim_n = outputevalarray(img_n,ref_y)
            psnr_p,ssim_p = outputevalarray(preds,ref_y)
            np.savetxt(self.resultpath/'eval'/(save_name+f'_psnr_{np.mean(psnr_re):.4f}.txt'), psnr_p, fmt='%.4f')
            np.savetxt(self.resultpath/'eval'/(save_name+f'_ssim_{np.mean(ssim_re):.6f}.txt'), ssim_p, fmt='%.6f')
            np.savetxt(self.resultpath/'eval'/(save_name+f'_psnr_n_{np.mean(psnr_in):.4f}.txt'), psnr_n, fmt='%.4f')
            np.savetxt(self.resultpath/'eval'/(save_name+f'_ssim_n_{np.mean(ssim_in):.6f}.txt'), ssim_n, fmt='%.6f')
            print(f"Name:{batch['id'][0]}, inputPSNR:{psnr_in:.4f}dB, outputPSNR:{psnr_re:.4f}dB, inputSSIM:{ssim_in:.6f}, outputSSIM:{ssim_re:.6f}.")
        # return (preds,psnr_val)

    def train_dataloader(self):
        # DataLoader class for training
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Define scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.patience,
            threshold = 0.01,verbose = True
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "val_psnr_mean",
            "name":'lr_monitor'
        }  # logged value to monitor
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs):
        # Calculate IOU at end of epoch
        val_psnr = sum(self.psnr_val)/len(self.psnr_val)
        #val_ssim = sum(self.psnr_val)/len(self.psnr_val)

        # Reset metrics before next epoch
        self.psnr_val = []
        self.ssim_val = []

        # Log epoch validation IOU
        self.log("val_psnr_mean", val_psnr, on_epoch=True, prog_bar=True, logger=True)
        return val_psnr


    ## Convenience Methods ##

    def _prepare_model(self):
        resnet = Resblock.__dict__['MultipleCascadeBlock_func']()
        #resnet = Resblock.__dict__['MultipleBasicBlock_4'](self.input_layers,self.hidden_layers,self.num_blocks)
        ####torch.nn.init.xavier_uniform_(resnet.weight.data)
        return resnet
        #cnn_denoise = torch.nn.Sequential(
        #torch.nn.Conv2d(self.input_layers, self.input_layers, kernel_size=5, stride=1,
        #             padding='same'),
        #torch.nn.ReLU()
        #)
        #torch.nn.init.normal_(cnn_denoise[0].weight.data, mean=0.0, std=1.0)
        #unet_model = smp.Unet(
        #    encoder_name='resnet34',
        #    encoder_weights='imagenet',
        #    in_channels=self.input_layers,
        #    classes=1,
        #)
        #s_stacked = torch.nn.Sequential(cnn_denoise, unet_model, torch.nn.Sigmoid())
        #if self.gpu:
        #    s_stacked.cuda()
        #return s_stacked

    def _get_trainer_params(self):
        # Define callback behavior
        lrmonitor_callback = pl.callbacks.LearningRateMonitor('epoch')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            monitor="val_psnr_mean",
            mode="max",
            verbose=True,
        )
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_psnr_mean",
            patience=(self.patience * 3),
            mode="max",
            verbose=True,
        )

        # Specify where TensorBoard logs will be saved
        self.log_path = Path.cwd() / self.hparams.get("log_path", "tensorboard-logs")
        self.log_path.mkdir(exist_ok=True)
        logger = pl.loggers.TensorBoardLogger(self.log_path, name="model-chasti")

        trainer_params = {
            "callbacks": [checkpoint_callback, early_stop_callback, lrmonitor_callback],
            "max_epochs": self.max_epochs,
            "min_epochs": self.min_epochs,
            "default_root_dir": self.output_path,
            "logger": logger,
            "gpus": None if not self.gpu else 1,
            "fast_dev_run": self.hparams.get("fast_dev_run", False),
            "num_sanity_val_steps": self.hparams.get("val_sanity_checks", 0),
        }
        return trainer_params

    def fit(self):
        # Set up and fit Trainer object
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.fit(self)

    def test(self):
        # Set up and fit Trainer object
        self.trainer = pl.Trainer(**self.trainer_params)
        self.trainer.test(self)
