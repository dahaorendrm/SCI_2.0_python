import torch
import numpy as np
from pathlib import Path

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from . import Resblock
from .loss import *
from utils import *

class SpViDeCNN(pl.LightningModule):
    def __init__(self, hparams):
        super(SpViDeCNN, self).__init__()
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
        self.val_psnr = []
        self.val_ssim = []
        self.val_psnr_incr = []
        self.val_ssim_incr = []

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
        img = batch['img'].float()
        img = torch.moveaxis(img,-1,1)
        img_n = batch['img_n'].float()
        img_n = torch.moveaxis(img_n,-1,1)
        sigma = batch['sigma'].float()
        criterion = XEDiceLoss()
        if self.gpu:
            img_n = img_n.cuda(non_blocking=True)
            img = img.cuda(non_blocking=True)
            sigma = sigma.cuda()
        sigma = sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        sigma = sigma.repeat(1, 1, img_n.size()[-2], img_n.size()[-1])
        input = torch.cat([img_n,sigma],1)
        pred = self.model(input[:,:8,...])
        loss = criterion(pred, img)
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
        img = batch['img'].float()
        img = torch.moveaxis(img,-1,1)
        img_n = batch['img_n'].float()
        img_n = torch.moveaxis(img_n,-1,1)
        sigma = batch['sigma'].float()
        #criterion = XEDiceLoss()
        if self.gpu:
            img_n = img_n.cuda(non_blocking=True)
            img = img.cuda(non_blocking=True)
            sigma = sigma.cuda()
        sigma = sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        sigma = sigma.repeat(1, 1, img_n.size()[-2], img_n.size()[-1])
        input = torch.cat([img_n,sigma],1)
        print(input.size())
        pred = self.model(input[:,:8,...])
        #loss = criterion(pred, img)

        img = img.cpu().numpy()
        img = np.squeeze(np.moveaxis(img,0,-1))
        img_n = img_n.cpu().numpy()
        img_n = np.squeeze(np.moveaxis(img_n,0,-1))
        pred = pred.cpu().numpy()
        pred = np.squeeze(np.moveaxis(pred,0,-1))

        val_psnr_input = calculate_psnr(img, img_n)
        val_ssim_input = calculate_ssim(img, img_n)
        val_psnr = calculate_psnr(img, pred)
        val_ssim = calculate_ssim(img, pred)
        val_psnr_incr = val_psnr - val_psnr_input
        val_ssim_incr = val_ssim - val_ssim_input
        self.val_psnr.append(val_psnr)
        self.val_ssim.append(val_ssim)
        self.val_psnr_incr.append(val_psnr_incr)
        self.val_ssim_incr.append(val_ssim_incr)

        self.log(
            "val_psnr_step_in",
            val_psnr_input,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_psnr_step",
            val_psnr,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        return (pred,val_psnr)

    def test_step(self, batch, batch_idx):
        self.model.eval()
        torch.set_grad_enabled(False)
        save_name = batch['id'][0]
        img = batch['img'].float()

        img_n = batch['img_n'].float()
        img_n = torch.moveaxis(img_n,-1,1)
        sigma = batch['sigma'].float()
        #criterion = XEDiceLoss()
        if self.gpu:
            img_n = img_n.cuda(non_blocking=True)

            sigma = sigma.cuda()
        sigma = sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        sigma = sigma.repeat(1, 1, img_n.size()[-2], img_n.size()[-1])
        input = torch.cat([img_n,sigma],1)
        pred = self.model(input[:,:8,...])
        img_n = img_n.cpu().numpy()
        img_n = np.squeeze(np.moveaxis(img_n,(0,1),(-1,-2)))
        pred = pred.cpu().numpy()
        pred = np.squeeze(np.moveaxis(pred,(0,1),(-1,-2)))
        if not img is False:
            img = torch.moveaxis(img,-1,1)
            img = img.cuda(non_blocking=True)
            #loss = criterion(pred, img)
            img = img.cpu().numpy()
            img = np.squeeze(np.moveaxis(img,(0,1),(-1,-2)))
            val_psnr_input = calculate_psnr(img, img_n)
            val_ssim_input = calculate_ssim(img, img_n)
            val_psnr = calculate_psnr(img, pred)
            val_ssim = calculate_ssim(img, pred)
        tifffile.imwrite(self.resultpath/f"{batch['id'][0]}_in.tiff",img_n)
        tifffile.imwrite(self.resultpath/f"{batch['id'][0]}.tiff",pred)
        print(f"Name:{save_name}, \
        inputPSNR:{val_psnr_input:.4f}dB, outputPSNR:{val_psnr:.4f}dB,\
        inputSSIM:{val_ssim_input:.6f}, outputSSIM:{val_ssim:.6f}.")


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
            num_workers=self.num_workers,
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
            "monitor": "val_psnr_epoch",
            "name":'lr_monitor'
        }  # logged value to monitor
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs):
        # Calculate IOU at end of epoch
        val_psnr = sum(self.val_psnr)/len(self.val_psnr)
        val_ssim = sum(self.val_ssim)/len(self.val_ssim)
        self.val_psnr = []
        self.val_ssim = []

        val_psnr_incr = sum(self.val_psnr_incr)/len(self.val_psnr_incr)
        val_ssim_incr = sum(self.val_ssim_incr)/len(self.val_ssim_incr)
        self.val_psnr_incr = []
        self.val_ssim_incr = []

        # Log epoch validation IOU
        self.log("val_psnr_epoch", val_psnr, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_ssim_epoch", val_ssim, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_psnr_change_epoch", val_psnr_incr, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_ssim_change_epoch", val_ssim_incr, on_epoch=True, prog_bar=True, logger=True)
        return val_psnr


    ## Convenience Methods ##

    def _prepare_model(self):
        Resblock.__dict__['MultipleBasicBlock2'](inplanes=8, intermediate_feature=128)

        return Resblock


    def _get_trainer_params(self):
        # Define callback behavior
        lrmonitor_callback = pl.callbacks.LearningRateMonitor('epoch')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            monitor="val_psnr_epoch",
            mode="max",
            verbose=True,
        )
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_psnr_epoch",
            patience=(self.patience * 3),
            mode="max",
            verbose=True,
        )

        # Specify where TensorBoard logs will be saved
        self.log_path = Path.cwd() / self.hparams.get("log_path", "tensorboard-logs")
        self.log_path.mkdir(exist_ok=True)
        logger = pl.loggers.TensorBoardLogger(self.log_path, name="model-SpViDeCNN")

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
