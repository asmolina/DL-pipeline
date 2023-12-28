import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _WeightedLoss
from typing import Callable, Optional
from torch import Tensor
from torchmetrics import (
    MetricCollection, 
    Accuracy, 
    Precision, 
    Recall, 
    ConfusionMatrix,
    F1Score,
    Dice
)
from torchmetrics.wrappers import ClasswiseWrapper

from torch.nn import CrossEntropyLoss
import wandb
import numpy as np

import segmentation_models_pytorch as smp
from models import UNet


class SegModule(pl.LightningModule):
    def __init__(
        self,
        loss_fn = None,
        model: str = 'unet',
        backbone: str = 'vgg19',
        optimizer: str = 'default',
        scheduler: str = 'default',
        lr: float = 3e-4,
        num_classes: int = 2,
        num_to_log: int = 16,
        step_size: int = 50,
        gamma: float = 0.1,
        in_channels: int =10,
        min_channels: int = 32,
        max_channels: int = 512,
        num_down_blocks: int = 4,
        ignore_index: int = 255, 
        class_labels_dict: dict = {
            0: 'лиственные',
            1: 'хвойные',
            2: 'смешанный',
            3: 'кустарник',
            255: 'нет_данных',
            }, # по возрастанию индекса
        labels_to_calc_metric: list = [
            'лиственные', 
            'хвойные', 
            'смешанные'
            ],
        possible_classes: list = [
            0, 1, 2
        ],
        activation: str = 'identity',
    ):
        # super(SegModule, self).__init__()
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.num_down_blocks = num_down_blocks
        self.ignore_index = ignore_index

        self.class_labels_dict = class_labels_dict
        self.possible_classes = possible_classes
        
        self.labels_to_calc_metric = labels_to_calc_metric  # по возрастанию индекса


        if model == 'unet':
            self.net = UNet(
                self.num_classes, 
                in_channels=self.in_channels, 
                min_channels=self.min_channels,
                max_channels=self.max_channels,
                num_down_blocks=self.num_down_blocks,
        )
        elif model == 'smp.Unet':
            self.net = smp.Unet(
                    encoder_name=backbone,        
                    encoder_depth=5, 
                    encoder_weights=None,     
                    in_channels=self.in_channels, 
                    activation=activation,  
                    classes=self.num_classes,                       
                )
        elif model == 'smp.Unet++':
            self.net = smp.UnetPlusPlus(
                    encoder_name=backbone,        
                    encoder_depth=5, 
                    encoder_weights=None,     
                    in_channels=self.in_channels, 
                    activation=activation,  
                    classes=self.num_classes,                       
                )
        elif model == 'smp.DeepLabV3+':
            self.net = smp.DeepLabV3Plus(
                    encoder_name=backbone,        
                    encoder_depth=5, 
                    encoder_weights=None,     
                    in_channels=self.in_channels, 
                    activation=activation,  
                    classes=self.num_classes,                       
                )


        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.eps = 1e-7
        self.step_size = step_size
        self.gamma = gamma
        
        # self.loss_fn = CrossEntropyLoss_notmasked(ignore_index=self.ignore_index) # CrossEntropyLoss_masked(ignore_index=255)
        self.loss_fn = CrossEntropyLoss(ignore_index=self.ignore_index) # Dice(ignore_index=self.ignore_index) 
        self.num_to_log = num_to_log

        metric_params = {
            'task': 'multiclass',
            'num_classes': self.num_classes,
            'ignore_index': self.ignore_index, 
        }

        ###### TRAIN
        self.train_precision = ClasswiseWrapper(
            Precision(average=None, **metric_params), 
            labels=self.labels_to_calc_metric,
            prefix='train/precision_',
            )
        self.train_recall = ClasswiseWrapper(
            Recall(average=None, **metric_params), 
            labels=self.labels_to_calc_metric,
            prefix='train/recall_',
            )
        self.train_f1score = ClasswiseWrapper(
            F1Score(average=None, **metric_params), 
            labels=self.labels_to_calc_metric,
            prefix='train/f1score_',
            )
        self.train_mean_acc = Accuracy(average='macro', **metric_params)
        


        ###### VALIDATION
        self.val_conf_matrix = ConfusionMatrix(**metric_params)
        self.val_precision = ClasswiseWrapper(
            Precision(average=None, **metric_params),
            labels=self.labels_to_calc_metric,
            prefix='val/precision_',
            )
        self.val_recall = ClasswiseWrapper(
            Recall(average=None, **metric_params),
            labels=self.labels_to_calc_metric,
            prefix='val/recall_',
            )
        self.val_f1score = ClasswiseWrapper(
            F1Score(average=None, **metric_params), 
            labels=self.labels_to_calc_metric,
            prefix='val/f1score_',
            )
        self.val_mean_acc = Accuracy(average='macro', **metric_params)
        
        self.val_step_imgs = []
        self.val_step_logits = []
        self.val_step_masks = []
        



        #### TEST
        self.test_conf_matrix = ConfusionMatrix(**metric_params)
        self.test_precision = ClasswiseWrapper(
            Precision(average=None, **metric_params),
            labels=self.labels_to_calc_metric,
            prefix='test/precision_',
            )
        self.test_recall = ClasswiseWrapper(
            Recall(average=None, **metric_params),
            labels=self.labels_to_calc_metric,
            prefix='test/recall_',
            )
        self.test_f1score = ClasswiseWrapper(
            F1Score(average=None, **metric_params), 
            labels=self.labels_to_calc_metric,
            prefix='test/f1score_',
            )
        self.test_mean_acc = Accuracy(average='macro', **metric_params)

        self.test_step_imgs = []
        self.test_step_logits = []
        self.test_step_masks = []
        

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        imgs, mask, _ = batch 
        logits = self.forward(imgs) 
        # logits.shape = torch.Size([batch_size, num_classes, 256, 256]) # torch.float32
        loss = self.loss_fn(logits, torch.squeeze(mask.long(), dim=1))
        
        # loss будет логироваться автоматически, метрики нужно логировать отдельно
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_mean_acc.update(logits, mask)
        self.train_precision.update(logits, mask)
        self.train_recall.update(logits, mask)
        return loss

    def on_training_epoch_end(self, outs):
        train_precision_epoch = self.train_precision.compute()
        train_recall_epoch = self.train_recall.compute()
        train_f1_epoch = self.train_f1score.compute()
        train_mean_acc_epoch = self.train_mean_acc.compute()

        self.log_dict({
            'train/mean_acc': train_mean_acc_epoch,
            **train_precision_epoch,
            **train_recall_epoch,
            **train_f1_epoch,
        })

        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1_epoch.reset()
        self.train_mean_acc.reset()



    def validation_step(self, batch, batch_idx):
        imgs, mask, _ = batch
        logits = self.forward(imgs)
        
        val_loss = self.loss_fn(logits, torch.squeeze(mask.long(), dim=1))

        self.val_step_imgs.append(imgs)
        self.val_step_logits.append(logits)
        self.val_step_masks.append(mask)

        self.val_precision.update(logits, mask)
        self.val_recall.update(logits, mask)
        self.val_f1score.update(logits, mask)
        self.val_mean_acc.update(logits, mask)
        

        self.log('val/loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
        res = {
            'img': imgs, 
            'logits': logits, 
            'mask': mask,
        }

        return res

    def on_validation_epoch_end(self):
        val_precision_epoch = self.val_precision.compute()
        val_recall_epoch = self.val_recall.compute()
        val_f1score_epoch = self.val_f1score.compute()
        val_mean_acc_epoch = self.val_mean_acc.compute()

        self.log_dict({
            'val/mean_acc': val_mean_acc_epoch,
            **val_precision_epoch,
            **val_recall_epoch,
            **val_f1score_epoch,
        })

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1score.reset()
        self.val_mean_acc.reset()

        all_imgs = torch.cat(self.val_step_imgs)
        all_logits = torch.cat(self.val_step_logits)
        all_masks = torch.cat(self.val_step_masks)
        

        # import matplotlib.pyplot as plt
        # print(all_imgs.shape)
        # print(all_imgs[0].shape)
        # print(all_masks[0].shape)
        # fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(8, 20))
        # for i in range(5):
        #     ax = axs[i // 5][0]
        #     ax.imshow(all_imgs[i][0, :, :].numpy(force=True))
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_aspect('auto')

        #     ax = axs[i // 5][1]
        #     ax.imshow(all_masks[i].numpy(force=True))
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_aspect('auto')
        # plt.show()

        self._log_images(all_imgs, all_logits, all_masks, 'val')

        self.val_step_imgs.clear()
        self.val_step_logits.clear()
        self.val_step_masks.clear()


    def test_step(self, batch, batch_idx):
        imgs, mask, _ = batch
        logits = self.forward(imgs)

        self.test_step_imgs.append(imgs)
        self.test_step_logits.append(logits)
        self.test_step_masks.append(mask)

        self.test_precision.update(logits, mask)
        self.test_recall.update(logits, mask)
        self.test_f1score.update(logits, mask)
        self.test_mean_acc.update(logits, mask)
    

    def on_test_epoch_end(self):
        test_precision_epoch = self.test_precision.compute()
        test_recall_epoch = self.test_recall.compute()
        test_f1score_epoch = self.test_f1score.compute()
        test_mean_acc_epoch = self.test_mean_acc.compute()

        self.log_dict({
            'test/mean_acc': test_mean_acc_epoch,
            **test_precision_epoch,
            **test_recall_epoch,
            **test_f1score_epoch,
        })

        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1score.reset()
        self.test_mean_acc.reset()

        all_imgs = torch.cat(self.test_step_imgs)
        all_logits = torch.cat(self.test_step_logits)
        all_masks = torch.cat(self.test_step_masks)

        self._log_images(all_imgs, all_logits, all_masks, 'test')

        self.test_step_imgs.clear()
        self.test_step_logits.clear()
        self.test_step_masks.clear()



    def _log_images(self, epoch_imgs, epoch_logits, epoch_masks, prefix):
        mask_list = []
        mask_list_B05 = []
        mask_list_B06 = []
        mask_list_B07 = []
        mask_list_B08 = []
        mask_list_B8A = []
        mask_list_B11 = []
        mask_list_B12 = []

        for i in range(self.num_to_log):

            cur_img = epoch_imgs[i]
            cur_logits = epoch_logits[i]
            cur_mask = epoch_masks[i]
            
            cur_pred_mask = torch.argmax(cur_logits, dim=0).numpy(force=True)
            cur_gt_mask = cur_mask.numpy(force=True)
            
            highlight_significant_pixels = np.vectorize(lambda x: x in set(self.possible_classes))
            cur_selected_gt_mask = highlight_significant_pixels(cur_gt_mask)
            correct_mask = np.where((cur_gt_mask == cur_pred_mask), 1, 0)
            correct_mask = np.where(cur_selected_gt_mask, correct_mask, 255)

            cur_img = cur_img.numpy(force=True)
            cur_img = np.round((cur_img + 1) * 255 / 2) # [-1:1] to [0:2] to [0.0:255.0] во всех каналах
            cur_rgb_img = cur_img[[2, 1, 0], :, :].transpose(1, 2, 0)

            masks_to_log = {
                    "predictions": {
                        "mask_data": cur_pred_mask,
                        "class_labels": self.class_labels_dict
                    },
                    "ground_truth": {
                        "mask_data": cur_gt_mask,
                        "class_labels": self.class_labels_dict
                    },
                    "errors_and_corrects": {
                        "mask_data": correct_mask,
                        "class_labels": {0: 'ошибка', 1: 'верно', 255: 'нет_данных'}
                    },
                    }

            # mask_img = wandb.Image(cur_rgb_img, mode='RGB', masks=**masks_to_log)
            # mask_img_B06 = wandb.Image(cur_img[4, :, :].numpy(force=True), mode='L', masks=**masks_to_log)
            # mask_img_B07 = wandb.Image(cur_img[5, :, :].numpy(force=True), mode='L', masks=**masks_to_log)
            # mask_img_B08 = wandb.Image(cur_img[6, :, :].numpy(force=True), mode='L', masks=**masks_to_log)
            # mask_img_B8A = wandb.Image(cur_img[7, :, :].numpy(force=True), mode='L', masks=**masks_to_log)
            # mask_img_B11 = wandb.Image(cur_img[8, :, :].numpy(force=True), mode='L', masks=**masks_to_log)
            # mask_img_B12 = wandb.Image(cur_img[9, :, :].numpy(force=True), mode='L', masks=**masks_to_log)

            mask_list.append(wandb.Image(cur_rgb_img, mode='RGB', masks=masks_to_log))
            mask_list_B05.append(wandb.Image(cur_img[3, :, :], mode='L', masks=masks_to_log))
            mask_list_B06.append(wandb.Image(cur_img[4, :, :], mode='L', masks=masks_to_log))
            mask_list_B07.append(wandb.Image(cur_img[5, :, :], mode='L', masks=masks_to_log))
            mask_list_B08.append(wandb.Image(cur_img[6, :, :], mode='L', masks=masks_to_log))
            mask_list_B8A.append(wandb.Image(cur_img[7, :, :], mode='L', masks=masks_to_log))
            mask_list_B11.append(wandb.Image(cur_img[8, :, :], mode='L', masks=masks_to_log))
            mask_list_B12.append(wandb.Image(cur_img[9, :, :], mode='L', masks=masks_to_log))


            self.logger.experiment.log({f'imgs/{prefix}_RGB': mask_list})
            self.logger.experiment.log({f'imgs/{prefix}_B05': mask_list_B05})
            self.logger.experiment.log({f'imgs/{prefix}_B06': mask_list_B06})
            self.logger.experiment.log({f'imgs/{prefix}_B07': mask_list_B07})
            self.logger.experiment.log({f'imgs/{prefix}_B08': mask_list_B08})
            self.logger.experiment.log({f'imgs/{prefix}_B8A': mask_list_B8A})
            self.logger.experiment.log({f'imgs/{prefix}_B11': mask_list_B11})
            self.logger.experiment.log({f'imgs/{prefix}_B12': mask_list_B12})



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


class CrossEntropyLoss_masked(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(
        self, 
        weight: Optional[Tensor] = None, 
        size_average=None, 
        ignore_index: int = -100,
        reduce=None, 
        reduction: str = 'mean',
        label_smoothing: float = 0.0
        ) -> None:

        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor, scl_mask: Tensor) -> Tensor:

        masked_input = input.masked_fill_(scl_mask, ignore_index)
        masked_target = target.masked_fill_(torch.squeeze(scl_mask, dim=1), ignore_index)

        res = F.cross_entropy(
            masked_input,
            torch.squeeze(masked_target.long(), dim=1),
            weight=self.weight,
            ignore_index=self.ignore_index, 
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
            )
        return res


class CrossEntropyLoss_notmasked(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(
        self, 
        weight: Optional[Tensor] = None, 
        size_average=None, 
        ignore_index: int = -100,
        reduce=None, 
        reduction: str = 'mean',
        label_smoothing: float = 0.0
        ) -> None:

        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor, scl_mask: Tensor) -> Tensor:
        res = F.cross_entropy(
            input,
            torch.squeeze(target.long(), dim=1),
            weight=self.weight,
            ignore_index=self.ignore_index, 
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
            )
        return res
