import math
import random
import torch
from torch import nn
import torchvision.transforms as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FkDataset
import fk
import numpy as np


def log(*m):
    if DEBUG:
        print(*m)
        
def log_progress(epoch, batch_number, n_batches, loss):
    s = f"Epoch: {epoch} \t Batch: {batch_number}/{n_batches} \t"
    s += "\t".join(["{}_loss: {:.4f}".format(k, v) for k, v in loss.items()])
    print(s, end="\r")
    return

def plot_progress(epoch, x_hat, x, loss, **kwargs):
    fk.plot.show(x_hat.detach().cpu().numpy(), vmin=None, vmax=None)
    fk.plot.show(x.detach().cpu().numpy(), vmin=None, vmax=None)
    return

class Downsample:
    def __init__(self, size, mode="bicubic"):
        self.size = size
        self.mode = mode
    
    def __call__(self, x):
        return torch.nn.functional.interpolate(x, self.size)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Unflatten(nn.Module):
    def __init__(self, size=256, h=None, w=None):
        self.size = size
        self.h = h if h is not None else 1
        self.w = w if w is not None else 1
        super(Unflatten, self).__init__()
        
    def forward(self, input):
        return input.view(input.size(0), self.size, self.w, self.h)

class Elu(nn.Module):
    def forward(self, x):
        return nn.functional.elu(x)
    
class UNetConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, padding):
        super(UNetConvBlock, self).__init__()

        # add convolution 1
        self.add_module("conv1",
                        nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=int(padding)))
        self.add_module("relu1", nn.ReLU())

        # add convolution 2
        self.add_module("conv2",
                        nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=int(padding)))
        self.add_module("relu2", nn.ReLU())

    def forward(self, x):
        return super().forward(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super(UNetUpBlock, self).__init__()

        # upsample
        self.up = nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=4,
                                     stride=4)

        # add convolutions block
        self.conv_block = UNetConvBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        padding=padding)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, skip_connection):
        up = self.up(x)
        crop1 = self.center_crop(skip_connection, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

    
class UNet(nn.Module):
    def __init__(self, filters, hidden_dim, input_size):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Args:
            hyperpara,s (dh.learning.Hyperparams): todo
            unet_hyperparams (dh.learning.ConvNetHyperparams): stores the hyperparameters of the downsampling branch of the unet
        """
        super(UNet, self).__init__()
        
        depth = len(filters)
        
        # downsampling
        self.downsample = nn.ModuleList()
        in_channels = 3  # v, w, u
        for i in range(depth):
            log(filters[i])
            self.downsample.append(UNetConvBlock(in_channels=in_channels,
                                                 out_channels=filters[i],
                                                 padding=1))
            in_channels = filters[i]
        
        # latent
            self.latent_cont = nn.Conv2d(in_channels, in_channels, 1, 1)
            self.flatten = Flatten()
            self.latent_in = nn.Linear(input_size // 2, hidden_size)
            self.latent_out = nn.Linear(hidden_size, input_size // 2)
            self.unflatten = Unflatten(input_size // 2 , 1, 1)
        
        # upsample
        self.upsample = nn.ModuleList()
        out_filter = [3] + filters
        for i in reversed(range(1, depth)):
            log(filters[i])
            self.upsample.append(UNetUpBlock(in_channels=in_channels,
                                             out_channels=out_filter[i],
                                             padding=1))
            in_channels = out_filter[i]
            
        self.output = nn.Conv2d(in_channels, 3, 1, 1)
        return
    
    def get_loss(self, y_hat, y):
        return torch.sqrt(F.mse_loss(y_hat, y, reduction="sum"))
    
    def forward(self, X):
        log("input", X.shape)
        skip_connections = []
        y_hat = X
        for i, down in enumerate(self.downsample):
            y_hat = down(y_hat)
            log("down", y_hat.shape)
            if i != len(self.downsample) - 1:
                skip_connections.append(y_hat)
                y_hat = F.max_pool2d(y_hat, 4)
                
        y_hat = self.flatten(y_hat)
        log(y_hat.shape)
        y_hat = self.latent_in(y_hat)
        log(X.shape)
        y_hat = self.latent_out(y_hat)
        log(y_hat.shape)
        y_hat = self.unflatten(y_hat)
        log(y_hat.shape)

        for i, up in enumerate(self.upsample):
            y_hat = up(y_hat, skip_connections[-i - 1])
            log("up", y_hat.shape)
        y_hat = self.output(y_hat)
        log("output", y_hat.shape)
        
        return X, self.get_loss(y_hat, X)
    
    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())
    
    
if __name__ == "__main__":
    ## HYPERPARAMS
    ROOT = "/home/ep119/repos/fenton_karma_jax/data/train_dev_set/"
    EPOCHS = 100000
    DEVICE = torch.device("cuda")
    INPUT_SIZE = 256
    HIDDEN_SIZE = 256
    BATCH_SIZE = 32
    DEBUG = False
    
    net = UNet([8, 16, 32, 64, 128], hidden_size, input_size).to(DEVICE)
    log(net)
    log(net.parameters_count())
    fkset = FkDataset(ROOT, 1, 0, 1, transforms=t.Compose([Downsample((INPUT_SIZE, INPUT_SIZE))]), squeeze=True)
    loader = DataLoader(fkset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0 if DEBUG else 12)
    optimiser = torch.optim.Adam(net.parameters())
    
    ## OPTIMISE
    for e in range(epochs):
        for b, y in enumerate(loader):
            y_hat, loss = net(y.to(DEVICE))
            
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            log_progress(e, b, len(loader), {"mse": loss})
            idx = random.randint(0, len(y_hat) - 1)
#         plot_progress(e, y_hat[idx], y[idx], loss)
        