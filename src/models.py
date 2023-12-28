import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import numpy as np

#########################################################################################
#########################################################################################
###########################           UNET         ######################################
#########################################################################################
#########################################################################################


class UNet(nn.Module):
    """
    A standard UNet network (with padding in covs).

    For reference, see the scheme in materials/unet.png
    - Use batch norm between conv and relu
    - Use max pooling for downsampling
    - Use conv transpose with kernel size = 3, stride = 2, padding = 1, and output padding = 1 for upsampling
    - Use 0.5 dropout after concat

    Args:
    - num_classes: number of output classes
    - min_channels: minimum number of channels in conv layers
    - max_channels: number of channels in the bottleneck block
    - num_down_blocks: number of blocks which end with downsampling

    The full architecture includes downsampling blocks, a bottleneck block and upsampling blocks

    You also need to account for inputs which size does not divide 2**num_down_blocks:
    interpolate them before feeding into the blocks to the nearest size which divides 2**num_down_blocks,
    and interpolate output logits back to the original shape
    """

    def __init__(
        self,
        num_classes,
        in_channels=10,
        min_channels=32,
        max_channels=512,
        num_down_blocks=4,
        ):
        # super(UNet, self).__init__()
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.num_down_blocks = num_down_blocks

        # 0. Zero_layer maps the 3 channels of the input image to
        # the minimum number of channels in downsampling block.
        self.zero_layer = nn.Conv2d(self.in_channels, self.min_channels, kernel_size=1, stride=1, padding=0)
        
        # 1. Downsampling
        # Each downsampling block doubles the number of chanells starting from min_channels if possible
        # (curent_num_channels * 2 <= max.channels), else it does not change the number of channels.
        in_channels_down = np.zeros(self.num_down_blocks + 1, dtype=int)
        out_channels_down = np.zeros(self.num_down_blocks + 1, dtype=int)
        in_channels_down[1] = self.min_channels
        out_channels_down[1] = self.min_channels * 2
        
        for i in range(2, self.num_down_blocks + 1):
            in_channels_down[i] = out_channels_down[i - 1]
            if 2 * in_channels_down[i] <= self.max_channels:
                out_channels_down[i] = 2 * in_channels_down[i]
            else:
                out_channels_down[i] = in_channels_down[i]
        # The last layer always has n_out = self.max_channels
        out_channels_down[-1] = self.max_channels
        
        # Adding all (num_down_blocks) downsampling blocks (the first one is `the closest` to the input image).
        for i in range(1, num_down_blocks + 1):
            n_in = in_channels_down[i]
            n_out = out_channels_down[i]
            block = nn.Sequential(
                    nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(n_out),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(n_out),
                    nn.ReLU(inplace=False),
                )
            setattr(self, "downsampling" + str(i), block)

            max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # max pooling for downsampling
            setattr(self, "downsampling_maxpool" + str(i), max_pool)
            
            
        # 2. Bottleneck
        # Adding the bottleneck.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.max_channels, self.max_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.max_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.max_channels, self.max_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.max_channels),
            nn.ReLU(inplace=False),
        )

                
        # 3. Upsampling
        # Adding several (num_down_blocks) upsampling blocks (the first one is `the closest` to the output image).
        for i in range(num_down_blocks, 0, -1):
            n_in = out_channels_down[i]
            n_out = in_channels_down[i]
            conv_transpose = nn.ConvTranspose2d(n_in, n_in, kernel_size=3, stride=2, padding=1, output_padding=1)  # conv transpose for upsampling
            setattr(self, "upsampling_conv" + str(i), conv_transpose)
    
            block = nn.Sequential(
                nn.Dropout2d(p=0.5, inplace=True),
                nn.Conv2d(out_channels_down[i] + n_in, n_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(n_out),
                nn.ReLU(inplace=False),
                nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(n_out),
                nn.ReLU(inplace=False),
            )
            setattr(self, "upsampling" + str(i), block)

        # 4. Conv 1x1 + softmax
        # Last_layer
        self.last_layer = nn.Sequential(
            nn.Conv2d(self.min_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.Sigmoid(),  ####################### Не должно быть такого
        )

    ############################################################################## 
    def forward(self, inputs):

        # Finding the nearest size which is divided by 2**num_down_blocks:
        new_input_width = self.find_nearest_size(inputs.shape[-1])
        new_input_height = self.find_nearest_size(inputs.shape[-2])

        # Interpolate inputs to the new size:
        if inputs.shape[-1] != new_input_width or inputs.shape[-2] != new_input_height:
            x = F.interpolate(inputs, size=(new_input_height, new_input_width), mode='bilinear', align_corners=False)
        else:
            x = inputs

        # 0. Mapping 3 -> self.min_channels
        x = self.zero_layer(x)

        # 1. Downsampling
        downsampling_outputs = {}
        for i in range(1, self.num_down_blocks + 1):
            # Go through downsampling block and save the result (will be used in upsampling block
            downsampling = getattr(self, "downsampling" + str(i))
            x = downsampling(x)
            downsampling_outputs[str(i)] = torch.clone(x)
            # Go through MaxPooling
            x = getattr(self, "downsampling_maxpool" + str(i))(x)

        # 2. Bottleneck
        x = self.bottleneck(x)

        # 3. Upsampling
        for i in range(self.num_down_blocks, 0, -1):
            # Go through conv transpose
            x = getattr(self, "upsampling_conv" + str(i))(x)
            # Concatenate the result and the tensor saved during downsampling block
            x = torch.cat((x, downsampling_outputs[str(i)]), dim=1)
            # Go through upsampling block
            x = getattr(self, "upsampling" + str(i))(x)

        # 4. Conv 1x1 + softmax
        logits = self.last_layer(x)

        # Interpolate logits back to the original size (if needed):
        if inputs.shape[-1] != new_input_width or inputs.shape[-2] != new_input_height:
            logits = F.interpolate(logits, size=(inputs.shape[-2], inputs.shape[-1]), mode='bilinear', align_corners=False)

        assert logits.shape == (
        inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits

    ############################################################################## 
    def find_nearest_size(self, input_size):
        p = 2 ** self.num_down_blocks
        k = 1
        while input_size > k * p:
            k += 1
        if abs(k * p - input_size) <= abs((k - 1) * p - input_size):
            return k * p
        else:
            return (k - 1) * p

