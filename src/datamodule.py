from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
import torch
import pytorch_lightning as pl
import albumentations.pytorch as AP
import albumentations as A
import numpy as np

def get_patches(im, patch_size, stride):
    im = torch.tensor(im)
    res = im.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    res = res.reshape(im.shape[0], -1, patch_size, patch_size)
    return res.numpy().transpose((1, 2, 3, 0))


class CustomDataset(Dataset):
    def __init__(
        self, 
        X, 
        Y, 
        SCL, 
        patch_size=256,
        stride=256, 
        transform=None, 
        skip_empty=True,
        max_percentage_of_nodata_pixels=0.6,
        ):
        self.X = get_patches(X, patch_size, stride) 
        self.Y = get_patches(Y, patch_size, stride)
        self.SCL = get_patches(SCL, patch_size, stride)

        self.transform = transform
        self.test_transform = A.Compose([
                AP.ToTensorV2(transpose_mask=True)
                ],
                additional_targets={'mask0':'mask'})

        self.num_of_patches = len(self.Y)
        self.patch_size = patch_size
        self.stride = stride
        self.total_height = X.shape[1] 
        self.total_width = X.shape[2] 
        
        print(f'🤖: Dataset initialized:') 
        if self.patch_size == self.stride:
            self.num_of_patches_in_row = self.total_width  // self.patch_size
            self.num_of_patches_in_col = self.total_height // self.patch_size
            print(f'🤖: ><><><><><><><><><> Total #patches: {self.num_of_patches} [{self.num_of_patches_in_col} x {self.num_of_patches_in_row}]')
            print(f'🤖: ><><><><><><><><><> {self.total_height % self.patch_size}px are left in vertical direction')
            print(f'🤖: ><><><><><><><><><> {self.total_width % self.patch_size}px are left in horizontal direction')
            print(f'🤖: ><><><><><><><><><> {self.Y.shape} | #patches x patch_size x patch_size x 1')

        if skip_empty:
            valid_indices = []
            for i, patch in enumerate(self.Y):
                # patch.shape = (128, 128, 1)
                if (patch == 255).sum() / (patch.shape[0] * patch.shape[1]) < max_percentage_of_nodata_pixels:
                    valid_indices.append(i)
            self.valid_indices = valid_indices
            print(f'🤖: ><><><><><><><><><> max_percentage_of_nodata_pixels = {max_percentage_of_nodata_pixels}')
            print(f'🤖: ><><><><><><><><><> {len(self.Y) - len(self.valid_indices)} patches were skipped')
            print(f'🤖: ><><><><><><><><><> {len(self.valid_indices)} are remaining')
        else:
            self.valid_indices = list(range(len(self.Y)))


    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        data = self.X[idx]
        label = self.Y[idx]
        scl = self.SCL[idx]

        if self.transform:
            t = self.transform(image=data, mask=label, mask0=scl)
        else:
            t = self.test_transform(image=data, mask=label, mask0=scl)

        data = t['image'].float()                   #### torch.float32    torch.Size([10, 512, 512])
        label = torch.squeeze(t['mask'], dim=0)     #### torch.int64      torch.Size([512, 512])
        scl = t['mask0'] != 4                       #### torch.bool       torch.Size([1,  512, 512])

        # img.shape = torch.Size([batch_size, 10, 256, 256]) # torch.float32
        # mask.shape = torch.Size([batch_size, 256, 256]) # torch.int64 
        # scl.shape = torch.Size([batch_size, 1, 256, 256])  # torch.bool
        
        return data, label, scl 



class GeoModule(pl.LightningDataModule):
    def __init__(
            self,
            X_train,
            Y_train, 
            X_test, 
            Y_test,
            SCL_train,
            SCL_test,
            transform=None,
            batch_size=16,
            patch_size=256,
            stride=256,
            num_workers=36, 
            norm_strategy='mean_std', # 'min_max' or 'mean_std'
            X_predict=np.array([])
            ):
        
        super().__init__()
        self.X_train = X_train.astype(float)
        self.X_test = X_test.astype(float)
        self.Y_train = Y_train.astype(int) ## нужно соблюсти type Long для target в crossEntropy
        self.Y_test = Y_test.astype(int)
        self.SCL_train = SCL_train
        self.SCL_test = SCL_test
        self.transform = transform
        self.batch_size = batch_size
        self.dataloader_params = {
            'batch_size': self.batch_size,
            'num_workers': num_workers,
            'drop_last': False,
        }
        self.patch_size = patch_size
        self.stride = stride
        self.norm_strategy = norm_strategy
        self.not_normalized = True
        self.X_predict = X_predict

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        print(f'🤖: Setup data...')
        if self.not_normalized and self.norm_strategy == 'mean_std':
            mean = self.X_train.mean(axis=(1, 2), keepdims=True)
            std = self.X_train.std(axis=(1, 2), keepdims=True)
            print(f'🤖: Mean = {mean}, std = {std}')
            self.X_train -= mean
            self.X_train /= std
            self.X_test -= mean
            self.X_test /= std
            if len(self.X_predict) > 0:
                self.X_predict -= mean
                self.X_predict /= std
            self.not_normalized = False

        elif self.not_normalized and self.norm_strategy == 'min_max':
            percentile_99 = np.percentile(self.X_train, q=99, axis=(1, 2), keepdims=True)
            percentile_1 = np.percentile(self.X_train, q=1, axis=(1, 2), keepdims=True)
            a = -1.0
            b = 1.0
            ### костыль
            c = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(-1, 1, 1) 
            d = np.array([7780., 7628., 7624., 7889., 7720., 7312., 7500., 6951., 4649., 4212.]).reshape(-1, 1, 1) 
            print(f'🤖: MANUALLY SELECTED PERCENTILES') 
        
            # c = percentile_1
            # d = percentile_99
            # print(f'🤖: percentile_99 = {percentile_99.flatten()}') 
            # print(f'🤖: percentile_1 = {percentile_1.flatten()}')

            self.X_train = (self.X_train - d) * (b - a) / (d - c) + a
            self.X_test = (self.X_test - d) * (b - a) / (d - c) + a
            if len(self.X_predict) > 0:
                self.X_predict = (self.X_predict - d) * (b - a) / (d - c) + a
            self.not_normalized = False

        if stage == "fit":
            ds_full = CustomDataset(
                self.X_train, 
                self.Y_train, 
                self.SCL_train, 
                patch_size=self.patch_size,
                stride=self.stride, 
                transform=self.transform, 
                skip_empty=True,
                max_percentage_of_nodata_pixels=0.6,
                )
            self.ds_train, self.ds_val = random_split(ds_full, [0.8, 0.2])

        if stage == "test":
            self.ds_test = CustomDataset(
                self.X_test, 
                self.Y_test, 
                self.SCL_test, 
                patch_size=self.patch_size, 
                stride=self.patch_size,
                transform=None, 
                skip_empty=False,
                )
        
        if stage == 'predict':
            self.ds_predict = PredictDataset(
                self.X_predict, 
                patch_height=self.patch_size,
                patch_width=self.patch_size, 
                overlap=self.patch_size // 4, 
                )

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
    

class PredictDataset(Dataset):
    def __init__(
        self, 
        X, 
        patch_height=256,
        patch_width=256, 
        overlap=64, 
        ):
        self.X, self.height_ind, self.width_ind = split_img(np.transpose(X, (1, 2, 0)), patch_height, patch_width, overlap) 

        self.test_transform = A.Compose([

                AP.ToTensorV2(transpose_mask=False)
                ],
                )        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = self.X[idx]
        t = self.test_transform(image=data)
        data = t['image'].float()                   #### torch.float32    torch.Size([10, 512, 512])
        return data



def split_img(img, patch_height, patch_width, overlap):
    
    max_height, max_width, num_bands = img.shape
    
    height_ind = 0
    height = 0
    imgarr = []

    while height < max_height:
        width_ind = 0
        width = 0
        while width < max_width:
            tmp = np.zeros((patch_height, patch_width, num_bands), dtype=img.dtype)
            
            tmp1_end_vertical = min(height + patch_height, max_height)
            tmp1_end_horizontal = min(width + patch_width, max_width)
            tmp1 = img[height:tmp1_end_vertical, width:tmp1_end_horizontal]
            tmp1_height, tmp1_width, _ = tmp1.shape
            
            tmp[:tmp1_height, :tmp1_width, :] = tmp1
            
            pad_width = tmp.shape[1] - tmp1_width
            pad_height = tmp.shape[0] - tmp1_height
            
            padded_tensor = np.pad(
                tmp1, 
                ((0, pad_height), (0, pad_width), (0, 0)), 
                'reflect',
                )

            imgarr.append(padded_tensor)

            width_ind += 1
            width += patch_width - 2 * overlap

        height += patch_height - 2 * overlap
        height_ind += 1
        
    return np.asarray(imgarr), height_ind, width_ind



def reconstruct_pred(pred, size_x, size_y, patch_height, patch_width, overlap, height_ind, width_ind):
    recon = np.empty((size_x, size_y, pred.shape[-1]), pred.dtype)
    final_patch = patch_height - overlap
    small_flag_y = (size_y < patch_width)

    for i in range(height_ind):
        for j in range(width_ind):
            recon[
                (i != 0) * (final_patch + (i - 1) * (patch_height - 2 * overlap)):min(
                    final_patch + i * (patch_height - 2 * overlap), 
                    size_x
                    ),
                (j != 0) * (final_patch + (j - 1) * (patch_width - 2 * overlap)): min(
                    final_patch + j * (patch_width - 2 * overlap), 
                    size_y), 
                    :
                    ] = pred[i * width_ind + j][overlap * (i != 0):-max(
                        overlap, 
                        final_patch - (size_x - (i != 0) * (final_patch + (i - 1) * (patch_height - 2 * overlap)))
                        ),
                overlap * (j != 0):-small_flag_y - max(
                    overlap, 
                    final_patch - (size_y - (j != 0) * (final_patch + (j - 1) * (patch_width - 2 * overlap)))), 
                    :
                ]
    return recon
