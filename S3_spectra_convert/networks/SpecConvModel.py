import torch
import numpy as np
from pathlib import Path

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import albumentations
import numpy as np
import tifffile
import os
from loss import *
import utils
import cv2

class SpecConvModel(pl.LightningModule):
    def __init__(self, hparams):
        super(SpecConvModel, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.backbone = self.hparams.get("backbone", "resnext50_32x4d")
        self.weights = self.hparams.get("weights", "imagenet")
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
        self.in_channels = self.hparams.get("in_channels", 2)
        self.out_channels = self.hparams.get("out_channels", 1)

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

    def training_step(self, batch, batch_idx):
        CUT_BAND = (4,2)
        # Switch on training mode
        self.model.train()
        torch.set_grad_enabled(True)

        # Load images and labels
        #print(f'shape chip:{batch["chip"].shape} nasadem:{batch["nasadem"].shape} recurrence:{batch["recurrence"].shape}')
        x = batch["feature"].float()
        y = batch["label"].float()
        #y = y[:,CUT_BAND[0]:-CUT_BAND[1],...]
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass
        preds = self.forward(x)

        # Calculate training loss
        criterion = XEDiceLoss()
        #print(f'shape of preds is {preds.shape}, shape of y is {y.shape}')
        loss = criterion(preds, y)

        # Log batch xe_dice_loss
        self.log(
            "loss",
            loss,
            #on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # Switch on validation mode
        CUT_BAND = (4,2)
        self.model.eval()
        torch.set_grad_enabled(False)

        # Load images and labels
        x = batch["feature"].float()
        y = batch["label"].float()
        #y = y[:,CUT_BAND[0]:-CUT_BAND[1],...]
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass & softmax
        preds = self.forward(x)
        #print(f'preds shape {preds.shape}')
        # preds = torch.softmax(preds, dim=1)[:, 1]

        # # Save validation output in temp
        # from PIL import Image
        # for i in range(preds.shape[0]):
        #     temp = np.squeeze(y.cpu().numpy()[i,...])
        #     #print(f'label squeezed y shape is {temp.shape}')
        #     Image.fromarray((temp*255).astype(np.uint8)).save(f"temp/vali{i}_true.jpg")
        #     Image.fromarray((np.squeeze(preds.cpu().numpy()[i,...])*255).astype(np.uint8)).save(f"temp/vali{i}_pred.jpg")

        # Calculate validation IOU (global)
        preds = preds.cpu().numpy()
        preds = np.squeeze(np.moveaxis(preds,1,-2))
        y = y.cpu().numpy()
        y = np.squeeze(np.moveaxis(y,1,-2))
        psnr_val = utils.calculate_psnr(preds,y)
        ssim_val = utils.calculate_ssim(preds,y)
        #psnr_val = utils.calculate_psnr(preds.to("cpu").numpy(),y.to("cpu").numpy())
        # ssim_val = utils.calculate_ssim(img_n,gt) %%% switch back the dimension
        self.psnr_val.append(psnr_val)
        self.ssim_val.append(ssim_val)

        # Log batch IOU
        self.log(
            'psnr',psnr_val,
            #'ssim',0,
            #on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True
        )
        self.log(
            'ssim',ssim_val,
            #'ssim',0,
            #on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True
        )
        return psnr_val

    def test_step(self, batch, batch_idx):
        # Switch on validation mode
        save_name = batch['id'][0].split('.')[0]
        CUT_BAND = (4,2)
        self.model.eval()
        torch.set_grad_enabled(False)

        # Load images and labels
        x = batch["feature"].float()
        if batch["label"] is not None:
            y = batch["label"].float()
            #y = y[:,CUT_BAND[0]:-CUT_BAND[1],...]
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass & softmax
        preds = []
        for i in range(x.size()[4]):
            x_temp = x[...,i]
            pred = self.forward(x_temp)
            preds.append(pred)
        #print(f'preds shape {preds.shape}')
        # preds = torch.softmax(preds, dim=1)[:, 1]

        # # Save validation output in temp
        # from PIL import Image
        # for i in range(preds.shape[0]):
        #     temp = np.squeeze(y.cpu().numpy()[i,...])
        #     #print(f'label squeezed y shape is {temp.shape}')
        #     Image.fromarray((temp*255).astype(np.uint8)).save(f"temp/vali{i}_true.jpg")
        #     Image.fromarray((np.squeeze(preds.cpu().numpy()[i,...])*255).astype(np.uint8)).save(f"temp/vali{i}_pred.jpg")

        # Calculate validation IOU (global)
        preds = torch.stack(preds,4)
        #print(f'shape of preds {preds.shape}, shape of y {y.shape}')
        psnr_val = None
        if batch["label"] is not None:
            preds = preds.cpu().numpy()
            preds = np.squeeze(np.moveaxis(preds,1,-2))
            y = y.cpu().numpy()
            y = np.squeeze(np.moveaxis(y,1,-2))
            psnr_val = utils.calculate_psnr(preds,y)
            ssim_val = utils.calculate_ssim(preds,y) # %%% switch back the dimension
            psnr_re,ssim_re = utils.outputevalarray(preds,y)
            np.savetxt(Path('result')/'eval'/(save_name+f'_psnr_{np.mean(psnr_re):.4f}.txt'), psnr_re, fmt='%.4f')
            np.savetxt(Path('result')/'eval'/(save_name+f'_ssim_{np.mean(ssim_re):.6f}.txt'), ssim_re, fmt='%.6f')
            print(f"Data {batch['id'][0]}, psnr : {psnr_val:.4f}, SSIM : {ssim_val:.6f}.")
            self.psnr_val.append(psnr_val)
            # self.ssim_val.append(ssim_val)
        #preds = torch.squeeze(preds)
        #preds = torch.moveaxis(preds,0,2)
        if not os.path.exists('./result/re'):
            os.mkdir('result')
            os.mkdir('result/re')
        tifffile.imwrite(f"result/re/{batch['id'][0]}",preds)
        utils.saveintemp(preds,save_name)
        utils.saveintemp(y,'orig'+save_name)
        #np.save(f'result/{batch["id"][0]}.npy',preds.cpu().numpy()) ####name needed
        # Log batch IOU
        self.log(
            'psnr',psnr_val,
            #'ssim',0,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
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
            batch_size=self.batch_size,
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
            "monitor": "val_ssim",
        }  # logged value to monitor
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs):
        # Calculate IOU at end of epoch
        val_psnr = sum(self.psnr_val)/len(self.psnr_val)
        val_ssim = sum(self.ssim_val)/len(self.ssim_val)
        
        # Reset metrics before next epoch
        self.psnr_val = []
        self.ssim_val = []

        # Log epoch validation IOU
        self.log("val_psnr", val_psnr, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_ssim", val_ssim, on_epoch=True, prog_bar=True, logger=True)
        #self.log("lr", self.learning_rate, on_epoch=True, prog_bar=True, logger=True)
        return val_psnr


    ## Convenience Methods ##

    def _prepare_model(self):
        cnn_denoise = torch.nn.Sequential(
        torch.nn.Conv2d(self.in_channels, self.in_channels, kernel_size=5, stride=1,
                     padding='same'),
        torch.nn.ReLU()
        )
        torch.nn.init.normal_(cnn_denoise[0].weight.data, mean=0.0, std=1.0)
        unet_model = smp.Unet(
            encoder_name=self.backbone,
            encoder_weights=self.weights,
            in_channels=self.in_channels,
            classes=self.out_channels,
        )
        s_stacked = torch.nn.Sequential(cnn_denoise, unet_model, torch.nn.Sigmoid())
        if self.gpu:
            s_stacked.cuda()
        return s_stacked

    def _get_trainer_params(self):
        # Define callback behavior
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.output_path,
            monitor="val_psnr",
            mode="max",
            verbose=True,
        )
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_psnr",
            patience=(self.patience * 3),
            mode="max",
            verbose=True,
        )

        # Specify where TensorBoard logs will be saved
        self.log_path = Path.cwd() / self.hparams.get("log_path", "tensorboard-logs")
        self.log_path.mkdir(exist_ok=True)
        logger = pl.loggers.TensorBoardLogger(self.log_path, name="benchmark-model")

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
