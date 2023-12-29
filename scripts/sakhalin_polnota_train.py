import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from pathlib import Path
import sys

SCRIPT_DIR = Path('/home/alina.smolina/eolearn-pipeline/src')
sys.path.append(str(SCRIPT_DIR))
SCRIPT_DIR = Path('/home/alina.smolina/DL-pipeline/src')
sys.path.append(str(SCRIPT_DIR))


eopatches_dir = '/beegfs/home/alina.smolina/data/sakhalin/images/EOPatches/train_2560_K_N_v4/'
num_ids_train = 598


eopatches_dir_test = '/beegfs/home/alina.smolina/data/sakhalin/images/EOPatches/test_2560_Nevelsk_v4/'
num_ids_test = 44


from datamodule_eolearn import GeoEOModule

import torch
torch.set_float32_matmul_precision('high')


import albumentations as A
import albumentations.pytorch as AP

transforms = A.Compose([
    A.Flip(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=(-0.0625, 0.0625), 
        scale_limit=0, #no scale
        rotate_limit=(-90, 90), 
        p=0.5
    ),
    AP.ToTensorV2(transpose_mask=True),
    ],
)

test_transform = A.Compose([
    AP.ToTensorV2(transpose_mask=True),
    ],
)


all_dates = GeoEOModule(
    transform = transforms,
    test_transform = test_transform,
    target_mask_name = 'polnota_code',
    train_eopatches_dir = eopatches_dir, 
    test_eopatches_dir = eopatches_dir_test, 
    predict_eopatches_dir = eopatches_dir_test, 
    train_eopatches_ids = [x for x in range(num_ids_train)], 
    test_eopatches_ids = [x for x in range(num_ids_test)],
    predict_eopatches_ids = [x for x in range(num_ids_test)], 
    # train_date_range = ['2018-01-01', '2018-12-31'],
    # test_date_range =['2018-01-01', '2018-12-31'],
    # predict_date_range = ['2018-01-01', '2018-12-31'], 
    train_date_range = ['2018-05-30', '2018-08-31'],
    test_date_range = ['2018-05-30', '2018-08-31'],
    predict_date_range = ['2018-05-30', '2018-08-31'],
    batch_size = 128,
    num_workers = 16,
)



import wandb
import pytorch_lightning as pl
from module import SegModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import os
os.environ["WANDB__SERVICE_WAIT"] = "500"

wandb_logger = WandbLogger(project='dl-pipeline-sakhalin-polnota', log_model=True)
print(f'🤖: Look at {wandb.run.url}')

lr_monitor_callback = LearningRateMonitor(logging_interval='step')


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=f'/beegfs/home/alina.smolina/DL-pipeline/weights/polnota-sakhalin/{wandb.run.name}/',
    filename='{epoch}-{val_loss:.4f}', 
    monitor='val/mean_acc',
    mode='max',
    save_top_k=1
)

trainer = pl.Trainer(
    max_epochs=1000, 
    benchmark=True, 
    check_val_every_n_epoch=10, 
    logger=wandb_logger, 
    callbacks=[
        checkpoint_callback,
        lr_monitor_callback,
            ],
)

model = SegModule(
    model='smp.DeepLabV3+', # 'smp.Unet++', #'smp.Unet',
    activation='identity',
    backbone='resnet50',
    optimizer='Adam', 
    scheduler='StepLR',
    step_size=200,
    gamma=0.8,
    lr=6e-3,
    in_channels=10,
    ignore_index=255,
    # min_channels = 16,
    # max_channels = 1024,
    # num_down_blocks = 6,
    num_classes=8, 
    class_labels_dict={
        0: 'полнота_0.3' ,
        1: 'полнота_0.4',
        2: 'полнота_0.5',
        3: 'полнота_0.6',
        4: 'полнота_0.7',
        5: 'полнота_0.8',
        6: 'полнота_0.9',
        7: 'полнота_1.0',
        255:'нет_данных',
    },
    labels_to_calc_metric=[
        'полнота_0.3', 
        'полнота_0.4', 
        'полнота_0.5',
        'полнота_0.6',
        'полнота_0.7',
        'полнота_0.8',
        'полнота_0.9',
        'полнота_1.0',
    ],
    possible_classes=[0, 1, 2, 3, 4, 5, 6, 7]
)

trainer.fit(model, all_dates) 

trainer.test(model, all_dates)

