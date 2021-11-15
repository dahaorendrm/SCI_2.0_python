import torch
import numpy as np
from pathlib import Path

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from . import Resblock
from loss import *
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
        mea = batch['mea'].float()
        if self.gpu:
            mea, y = mea.cuda(non_blocking=True), y.cuda(non_blocking=True)
        preds = []
        for i in range(batch['img_n'].shape[3]):
            img_n = batch['img_n'][...,i].float()
            oth_n = batch['oth_n'][...,i].float()
            mask = batch['mask'][...,i].float()
            if self.gpu:
                img_n, mask = img_n.cuda(non_blocking=True), mask.cuda(non_blocking=True)
            # print(f'shape of input is {torch.stack((mea,img_n,mask),1).size()}')
            pred = self.model(torch.stack((mea,img_n,mask,oth_n),1))
            preds.append(torch.squeeze(pred,1))
        preds = torch.stack(preds,3)
        criterion = XEDiceLoss()
        # print(f'shape of preds is {preds.size()}, shape of y is {y.size()}')
        loss = criterion(preds, self.selectFrames(y))
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
            mea, y = mea.cuda(non_blocking=True), y.cuda(non_blocking=True)
        preds = []
        for i in range(batch['img_n'].shape[3]):
            img_n = batch['img_n'][...,i].float()
            mask = batch['mask'][...,i].float()
            oth_n = batch['oth_n'][...,i].float()
            if self.gpu:
                img_n, mask = img_n.cuda(non_blocking=True), mask.cuda(non_blocking=True)
            pred = self.model(torch.stack((mea,img_n,mask),1))
            preds.append(torch.squeeze(pred,1))
        preds = torch.stack(preds,3)
        saveintemp(preds.cpu().numpy(),batch['id'][0])
        #print(f'shape of preds is {preds.size()}, label is {y.size()}')
        psnr_val_n = calculate_psnr(batch['img_n'].cpu().numpy(), y.cpu().numpy())
        psnr_val = calculate_psnr(preds.cpu().numpy(), y.cpu().numpy())
        ssim_val = calculate_ssim(preds.cpu().numpy(), y.cpu().numpy())
        self.psnr_val.append(psnr_val)
        self.log(
            "val_psnr_in",
            psnr_val_n,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_psnr",
            psnr_val,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_ssim",
            ssim_val,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return (preds,psnr_val)

    def test_step(self, batch, batch_idx):
        self.model.eval()
        torch.set_grad_enabled(False)
        mea = batch['mea']
        if self.gpu:
            mea, y = mea.cuda(non_blocking=True), y.cuda(non_blocking=True)
        preds = []
        for i in range(batch['img_n'].shape[3]):
            img_n = batch['img_n'][...,i].float()
            mask = batch['mask'][...,i].float()
            oth_n = batch['oth_n'][...,i].float()
            if self.gpu:
                img_n, mask = img_n.cuda(non_blocking=True), mask.cuda(non_blocking=True)
            pred = model(torch.stack((mea,img_n,mask),3))
            preds.append(torch.squeeze(pred,1))
        preds = torch.stack(preds,3)
        psnr_val = None
        if 'label' in batch.keys():
            y = batch['label']
            y = self.selectFrames(y)
            psnr_val = calculate_psnr(preds.cpu().numpy(), y.numpy())
            self.log(
                "psnr",
                psnr_val,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
        preds = torch.squeeze(preds)
        tifffile.imwrite(f'result/{batch["id"][0]}.tiff',preds.cpu().numpy()) ####name needed
        return (preds,psnr_val)

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
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def configure_optimizers(self):
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Define scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.patience
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "val_psnr_mean",
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
        self.log("lr", self.learning_rate, on_epoch=True, prog_bar=True, logger=True)
        return val_psnr


    ## Convenience Methods ##

    def _prepare_model(self):
        resnet = Resblock.__dict__['MultipleBasicBlock_4'](self.input_layers,self.hidden_layers,self.num_blocks)
        #torch.nn.init.xavier_uniform_(resnet.weight.data)
        return resnet

    def _get_trainer_params(self):
        # Define callback behavior
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
        logger = pl.loggers.TensorBoardLogger(self.log_path)

        trainer_params = {
            "callbacks": [checkpoint_callback, early_stop_callback],
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
