from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
import torch
import pytorch_lightning as pl
import albumentations.pytorch as AP
import albumentations as A
import numpy as np

import time

from eolearn.core import EOPatch, FeatureType
from datetime import date, datetime
from tqdm.autonotebook import tqdm

class EOLearnDataset(Dataset):
    def __init__(
        self, 
        transform = A.Compose([AP.ToTensorV2(transpose_mask=True)]),
        eopatches_dir = './EOPatches/', 
        eopatches_ids = [x for x in range(69, 106)],
        date_range = ['2018-01-01', '2018-12-31'],
        target_mask_name = 'PREVAIL_GROUP',
        target_feature_name = FeatureType.MASK_TIMELESS, # для классификации FeatureType.MASK_TIMELESS, для регрессии FeatureType.DATA_TIMELESS
        demand_target = True,
        ):

        self.demand_target = demand_target
        self.target_mask_name = target_mask_name
        self.target_feature_name = target_feature_name

        self.eopatches_dir = eopatches_dir
        self.eopatches_ids = eopatches_ids
        self.transform = transform

        date_range = [date.fromisoformat(d) for d in date_range]
        date_range = [datetime.combine(d, datetime.min.time()) for d in date_range]
        date1, date2 = date_range
        self.date_range = date_range

        print(f'🤖: Collecting all the time indices...', end=' ')
        st = time.time()
        count_timestamps = {}
        for eopatch_id in tqdm(self.eopatches_ids):
            eopatch = EOPatch.load(f'{self.eopatches_dir}eopatch_{eopatch_id}', lazy_loading=True)
            eopatch = eopatch.temporal_subset(lambda times: [date1 < t < date2 for t in times])
            count_timestamps[eopatch_id] = len(eopatch.timestamps)
        self.count_timestamps = count_timestamps

        self.cumsum = [0] 
        for v in self.count_timestamps.values():
            self.cumsum += [self.cumsum[-1] + v]
        et = time.time()
        print('Done.')

        print(f'🤖: Generating mapping: idx -> (patch_id, time_id)...', end=' ')
        self.patch_time_ids = []
        total_num_of_ids = sum(self.count_timestamps.values())
        for idx in tqdm(range(total_num_of_ids)):
            patch_id = self.get_patch_index(idx)
            time_id = idx - self.cumsum[patch_id]
            self.patch_time_ids.append((patch_id, time_id))
        
        print(f'Elapsed time: {((et - st) / 60):.2f} min', end=' ')
        print('Done.')

        print(f'🤖: Loading all the patches and time frames into the memory...', end=' ')
        st = time.time()
        self.bands = []
        self.labels = []
        # print('DUBUG', total_num_of_ids, self.patch_time_ids)
        # print('DUBUG', self.cumsum)

        for item in tqdm(self.patch_time_ids):
            patch_id, time_id = item
            eopatch = EOPatch.load(
                f'{self.eopatches_dir}eopatch_{self.eopatches_ids[patch_id]}', 
                lazy_loading=True, 
                features=[
                    (FeatureType.DATA, '10BANDS'), 
                    (self.target_feature_name, self.target_mask_name)
                    ]
                )
            eopatch = eopatch.temporal_subset(lambda times: [date1 < t < date2 for t in times])
            eopatch = eopatch.temporal_subset([time_id])
            data = eopatch[FeatureType.DATA]['10BANDS'].squeeze().astype(np.float32)
            self.bands.append(data)

            if self.demand_target:
                label = eopatch[self.target_feature_name][self.target_mask_name] ####.astype(np.uint8)
                self.labels.append(label)
        et = time.time()
        print('Done.')
        print(f'Elapsed time: {((et - st) / 60):.2f} min', end=' ')
        



    def get_patch_index(self, idx): # get patch index
        left, right = 0, len(self.cumsum) - 1
        while left < right:
            middle = (left + right + 1) // 2
            if self.cumsum[middle] > idx:
                right = middle - 1
            else:
                left = middle
        return left

    
        

    def __getitem__(self, idx):
        data = self.bands[idx]

        band_info = self.patch_time_ids[idx]

        if self.demand_target:
            label = self.labels[idx]
            t = self.transform(image=data, mask=label)
            data = t['image'].float()       #### torch.float32    torch.Size([10, 512, 512])
            label = t['mask'].squeeze()     #### torch.int64      torch.Size([512, 512])
        else:
            label = None
            # print('label is None')
            t = self.transform(image=data)
            data = t['image'].float() 
        
        return data, label, band_info  ### TO_DO: remove the last 'label'


    def __len__(self):
        return sum(self.count_timestamps.values())



class GeoEOModule(pl.LightningDataModule):
    def __init__(
            self,
            transform = A.Compose([AP.ToTensorV2(transpose_mask=True)]),
            test_transform = A.Compose([AP.ToTensorV2(transpose_mask=True)]), 
            target_mask_name = 'PREVAIL_GROUP',
            train_eopatches_dir = './EOPatches/', 
            test_eopatches_dir = './EOPatches/', 
            predict_eopatches_dir = './EOPatches/', 
            train_eopatches_ids = [x for x in range(69)],
            test_eopatches_ids = [x for x in range(69, 106)],
            predict_eopatches_ids = [x for x in range(106)], 
            train_date_range = ['2018-01-01', '2018-12-31'],
            test_date_range = ['2018-01-01', '2018-12-31'],
            predict_date_range = ['2018-01-01', '2018-12-31'],
            batch_size = 64,
            num_workers = 24, 
            target_feature_name = FeatureType.MASK_TIMELESS,
            ):
        
        super().__init__()
        self.target_mask_name = target_mask_name
        self.target_feature_name = target_feature_name
        self.transform = transform
        self.test_transform = test_transform

        self.train_eopatches_dir = train_eopatches_dir
        self.test_eopatches_dir = test_eopatches_dir
        self.predict_eopatches_dir = predict_eopatches_dir

        self.train_eopatches_ids = train_eopatches_ids
        self.test_eopatches_ids = test_eopatches_ids
        self.predict_eopatches_ids = predict_eopatches_ids

        self.train_date_range = train_date_range
        self.test_date_range = test_date_range
        self.predict_date_range = predict_date_range
        
        self.batch_size = batch_size
        self.dataloader_params = {
            'batch_size': self.batch_size,
            'num_workers': num_workers,
            'drop_last': False,
        }

        # self.not_normalized = True # Flag. Used to prevent to using for setup() twice
        

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        print(f'🤖: Setup data...')
        if stage == "fit":
            print(f'🤖: Fit stage.')
            ds_full = EOLearnDataset(
                transform = self.transform,
                eopatches_dir = self.train_eopatches_dir, 
                eopatches_ids = self.train_eopatches_ids,
                date_range = self.train_date_range,
                target_mask_name = self.target_mask_name,
                target_feature_name = self.target_feature_name,
                demand_target = True,
            )
            print(f'🤖: Fit stage.')
            self.ds_train, self.ds_val = random_split(ds_full, [0.8, 0.2])
            print(f'🤖: Fit frames >>> {len(self.ds_train)}')
            print(f'🤖: Val frames >>> {len(self.ds_val)}')
            print(f'🤖: Total >>>>>>>> {len(ds_full)}')

        if stage == "test":
            self.ds_test = EOLearnDataset(
                transform = self.test_transform,
                eopatches_dir = self.test_eopatches_dir, 
                eopatches_ids = self.test_eopatches_ids,
                date_range = self.test_date_range,
                target_mask_name = self.target_mask_name,
                target_feature_name = self.target_feature_name,
                demand_target = True,
                )
            print(f'🤖: Test stage.')
            print(f'🤖: Test frames >>> {len(self.ds_test)}')
        
        if stage == 'predict':
            self.ds_predict = EOLearnDataset(
                transform = self.test_transform,
                eopatches_dir = self.predict_eopatches_dir, 
                eopatches_ids = self.predict_eopatches_ids,
                date_range = self.predict_date_range,
                target_mask_name = self.target_mask_name,
                demand_target = False,
            )
            print(f'🤖: Predict stage.')
            print(f'🤖: Predict frames >>> {len(self.ds_predict)}')


    def train_dataloader(self):
        res = DataLoader(self.ds_train, shuffle=True, **self.dataloader_params)
        print(f'🤖: #iterations in train dataloader: {len(res)}')
        return res

    def val_dataloader(self):
        res = DataLoader(self.ds_val, **self.dataloader_params)
        print(f'🤖: #iterations in val dataloader: {len(res)}')
        return res

    def test_dataloader(self):
        res = DataLoader(self.ds_test, **self.dataloader_params)
        print(f'🤖: #iterations in test dataloader: {len(res)}')
        return res
    
    def predict_dataloader(self):
        res = DataLoader(self.ds_predict, **self.dataloader_params)
        print(f'🤖: #iterations in predict dataloader: {len(res)}')
        return res
    
    