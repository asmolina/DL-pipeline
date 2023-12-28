import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Callable, Optional
from torch import Tensor
from torchmetrics import (
    MetricCollection, 
    R2Score,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from torchmetrics.wrappers import ClasswiseWrapper
from torch.nn import MSELoss, L1Loss
from torch.nn.modules.loss import _WeightedLoss, _Loss

import segmentation_models_pytorch as smp
from models import UNet

import wandb
import numpy as np



class RegModule(pl.LightningModule):
    def __init__(
        self,
        loss_fn = None,
        model: str = 'unet',
        backbone: str = 'resnet34',
        optimizer: str = 'default',
        scheduler: str = 'default',
        lr: float = 3e-4,
        num_to_log: int = 16,
        step_size: int = 50,
        gamma: float = 0.1,
        in_channels: int =10,
        min_channels: int = 32,
        max_channels: int = 512,
        num_down_blocks: int = 4,
        ignore_index: int = 255, 
        activation: str = 'identity',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.num_down_blocks = num_down_blocks
        self.ignore_index = ignore_index


        if model == 'unet':
            self.net = UNet(
                1, 
                in_channels=self.in_channels, 
                min_channels=self.min_channels,
                max_channels=self.max_channels,
                num_down_blocks=self.num_down_blocks
                )
        elif model == 'smp.Unet':
            self.net = smp.Unet(
                    encoder_name=backbone,        
                    encoder_depth=5, 
                    encoder_weights=None,     
                    in_channels=self.in_channels, 
                    activation=activation,  
                    classes=1,                       
                )


        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.eps = 1e-7
        self.step_size = step_size
        self.gamma = gamma
        
        # self.loss_fn = MSELoss_ignoreindex(ignore_index=self.ignore_index) 
        self.loss_fn = L1Loss(reduction='mean') #MSELoss(reduction='mean') 

        self.num_to_log = num_to_log

        ###### TRAIN
        self.train_mape = MeanAbsolutePercentageError()
        self.train_mae = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        self.train_r2 = R2Score()

        ###### VALIDATION
        self.val_mape = MeanAbsolutePercentageError()
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_r2 = R2Score()

        self.val_step_imgs = []
        self.val_step_logits = []
        self.val_step_masks = []
        self.val_patch_info = []
        self.val_ignore_mask = []

        #### TEST
        self.test_mape = MeanAbsolutePercentageError()
        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()


        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()


    def forward(self, x):
        return self.net(x)


    def training_step(self, batch, batch_idx):
        imgs, mask, _ = batch 

        logits = self.forward(imgs)# logits.shape = torch.Size([batch_size, num_classes, 256, 256]) # torch.float32

        logits = torch.squeeze(logits, dim=1) # logits.shape = torch.Size([batch_size, 256, 256])
        mask = torch.squeeze(mask, dim=1)
        ignore_mask = mask == self.ignore_index
        masked_logits = logits[~ignore_mask]
        masked_target = mask[~ignore_mask]
        
        loss = self.loss_fn(masked_logits, masked_target)
        
        # loss будет логироваться автоматически, метрики нужно логировать отдельно
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.train_mape.update(masked_logits, masked_target)
        self.train_mae.update(masked_logits, masked_target)
        self.train_mse.update(masked_logits, masked_target)
        self.train_r2.update(masked_logits, masked_target)


    def on_training_epoch_end(self):
        train_mape_epoch = self.train_mape.compute()
        train_mae_epoch = self.train_mae.compute()
        train_mse_epoch = self.train_mse.compute()
        train_r2_epoch = self.train_r2.compute()

        self.log_dict({
            'train/mape': train_mape_epoch,
            'train/mae': train_mae_epoch,
            'train/mse': train_mse_epoch,
            'train/r2': train_r2_epoch, 
        })

        self.train_mape.reset()
        self.train_mae.reset()
        self.train_mse.reset()
        self.train_r2.reset()

############### VALIDATION
    def validation_step(self, batch, batch_idx):
        imgs, mask, band_info = batch 

        logits = self.forward(imgs)# logits.shape = torch.Size([batch_size, num_classes, 256, 256]) # torch.float32

        logits = torch.squeeze(logits, dim=1) # logits.shape = torch.Size([batch_size, 256, 256])
        mask = torch.squeeze(mask, dim=1)
        ignore_mask = mask == self.ignore_index
        masked_logits = logits[~ignore_mask]
        masked_target = mask[~ignore_mask]

        print('masked logits', masked_logits.shape, masked_logits)
        print('masked_target', masked_target.shape, masked_target)

        loss = self.loss_fn(masked_logits, masked_target)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.val_mape.update(masked_logits, masked_target)
        self.val_mae.update(masked_logits, masked_target)
        self.val_mse.update(masked_logits, masked_target)
        self.val_r2.update(masked_logits, masked_target)

        self.val_ignore_mask.append(ignore_mask)
        self.val_patch_info.append(band_info)
        self.val_step_imgs.append(imgs)
        self.val_step_logits.append(logits)
        self.val_step_masks.append(mask)


    def on_validation_epoch_end(self):
        val_mape_epoch = self.val_mape.compute()
        val_mae_epoch = self.val_mae.compute()
        val_mse_epoch = self.val_mse.compute()
        val_r2_epoch = self.val_r2.compute()

        self.log_dict({
            'val/mape': val_mape_epoch,
            'val/mae': val_mae_epoch,
            'val/mse': val_mse_epoch,
            'val/r2': val_r2_epoch, 
        })

        self.val_mape.reset()
        self.val_mae.reset()
        self.val_mse.reset()
        self.val_r2.reset()

        ######## DEBUG:
        all_imgs = torch.cat(self.val_step_imgs)
        all_logits = torch.cat(self.val_step_logits)
        all_masks = torch.cat(self.val_step_masks)
        all_ignore_masks = torch.cat(self.val_ignore_mask)
        
        import matplotlib.pyplot as plt
        print('Learning rate', self.lr)
        print(self.val_patch_info)
        print(all_imgs.shape)
        print(all_imgs[0].shape)
        print(all_masks[0].shape)
        nrows = all_imgs.shape[0]
        fig, axs = plt.subplots(nrows=nrows, ncols=4, figsize=(4*3, nrows*3))

        for i in range(nrows):
            ax = axs[i % nrows][0]
            ax.imshow(all_imgs[i][0, :, :].numpy(force=True))
            print(all_imgs[i][0, :, :].numpy(force=True))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

            ax = axs[i % nrows][1]
            ax.imshow(all_masks[i].numpy(force=True))
            print(all_masks[i].numpy(force=True))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

            ax = axs[i % nrows][2]
            ax.imshow(all_logits[i].numpy(force=True))
            print(all_logits[i].numpy(force=True))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')

            ax = axs[i % nrows][3]
            ax.imshow(all_ignore_masks[i].numpy(force=True))
            print(all_ignore_masks[i].numpy(force=True))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
        plt.show()

        self.val_patch_info.clear()
        self.val_step_imgs.clear()
        self.val_step_logits.clear()
        self.val_step_masks.clear()
        self.val_ignore_mask.clear()


############### TEST
    def test_step(self, batch, batch_idx):
        imgs, mask, _ = batch 

        logits = self.forward(imgs) # logits.shape = torch.Size([batch_size, num_classes, 256, 256]) # torch.float32

        logits = torch.squeeze(logits, dim=1) # logits.shape = torch.Size([batch_size, 256, 256])
        mask = torch.squeeze(mask, dim=1)
        ignore_mask = mask == self.ignore_index
        masked_logits = logits[~ignore_mask]
        masked_target = mask[~ignore_mask]

        self.test_mape.update(masked_logits, masked_target)
        self.test_mae.update(masked_logits, masked_target)
        self.test_mse.update(masked_logits, masked_target)
        self.test_r2.update(masked_logits, masked_target)


    def on_test_epoch_end(self):
        test_mape_epoch = self.test_mape.compute()
        test_mae_epoch = self.test_mae.compute()
        test_mse_epoch = self.test_mse.compute()
        test_r2_epoch = self.test_r2.compute()

        self.log_dict({
            'test/mape': test_mape_epoch,
            'test/mae': test_mae_epoch,
            'test/mse': test_mse_epoch,
            'test/r2': test_r2_epoch, 
        })

        self.test_mape.reset()
        self.test_mae.reset()
        self.test_mse.reset()
        self.test_r2.reset()




    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            opt = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        
        if self.scheduler == 'StepLR':
            sch = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=self.step_size, gamma=self.gamma)
        elif self.scheduler == 'LambdaLR':
            sch = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda x: 1)

        return [opt], [sch]



class MSELoss_ignoreindex(_Loss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(
        self, 
        ignore_index: int = 255,
        reduction: str = 'mean',
        size_average=None, 
        reduce=None,
        ) -> None:

        super().__init__(reduction)
        self.ignore_index = ignore_index

    def forward(
        self, 
        input: Tensor, 
        target: Tensor
        ) -> Tensor:

        modified_target = torch.squeeze(target, dim=1)
        ignore_mask = modified_target == self.ignore_index
        res = (input[~ignore_mask] - modified_target[~ignore_mask])**2

        if reduction == 'mean':
            return res.mean()
        elif reduction == 'None':
            return res

