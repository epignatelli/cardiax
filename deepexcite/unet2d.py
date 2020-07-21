import torch
import torch.nn.functional as F
from torch import nn


class UNetConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, padding, batch_norm):
        super(UNetConvBlock, self).__init__()

        # add convolution 1
        self.add_module("conv1",
                        nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=int(padding)))
        self.add_module("relu1", nn.ReLU())

        # add batchnorm 1
        if batch_norm:
            self.add_module("batchnorm1", nn.BatchNorm2d(out_channels))

        # add convolution 2
        self.add_module("conv2",
                        nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=int(padding)))
        self.add_module("relu2", nn.ReLU())

        # add batchnorm 2
        if batch_norm:
            self.add_module("batchnorm2", nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return super().forward(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding, batch_norm):
        super(UNetUpBlock, self).__init__()

        # upsample
        self.up = nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=2,
                                     stride=2)

        # add convolutions block
        self.conv_block = UNetConvBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        padding=padding,
                                        batch_norm=batch_norm)

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

    
class RecurrentUNet(nn.Module):
    def __init__(self, channels, frames_in, frames_out, step, attention=None):
        super(RecurrentUNet, self).__init__()

        in_channels = hyperparams.n_frames_in
        self.project_state = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        # downsampling
        self.downsample = nn.ModuleList()
        for i in range(len(channels)):
            self.downsample.append(UNetConvBlock(in_channels=in_channels,
                                                 out_channels=unet_hyperparams.filters_dimensions[i],
                                                 padding=unet_hyperparams.padding[i],
                                                 batch_norm=unet_hyperparams.batch_norm))
            in_channels = unet_hyperparams.filters_dimensions[i]

        # upsample
        self.upsample = nn.ModuleList()
        for i in reversed(range(unet_hyperparams.depth - 1)):
            self.upsample.append(UNetUpBlock(in_channels=in_channels,
                                             out_channels=unet_hyperparams.filters_dimensions[i],
                                             padding=unet_hyperparams.padding[i],
                                             batch_norm=unet_hyperparams.batch_norm))

            in_channels = unet_hyperparams.filters_dimensions[i]

        # return only one frame out
        self.output = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)

    def forward(self, X, Y=None, X_hidden=None):
        X = self.project_state(X)

        skip_connections = []
        for i, down in enumerate(self.downsample):
            X = down(X)
            if i != len(self.downsample) - 1:
                skip_connections.append(X)
                X = F.max_pool2d(X, 2)

        for i, up in enumerate(self.upsample):
            X = up(X, skip_connections[-i - 1])

        X_real = self.output(X)

        return X_real

    def training_step(self, batch, batch_idx):
        X = batch[:, :self.frames_in]  # 2 frames
        Y = batch[:, self.frames_in:]  # 20 output frames

        inputs = X
        y_hat, y_hidden = None, None
        for i in range(self.n_frames_out):
            x = inputs[:, -self.n_frames_in:]
            y = Y[:, i].unsqueeze(1)
            y_hat = self(x, y)
            inputs = torch.cat([X[:, -1].unsqueeze(1), y_hat], dim=1)
            loss.backward(retain_graph=True)
        optimiser.step()
        optimiser.zero_grad()

        return (y_hat, y_hidden), loss

    def validation_sep(self, batch, batch_idx):
        X = batch[:, :self.frames_in]  # 2 frames
        Y = batch[:, self.frames_in:]  # 20 output frames

        inputs = X
        y_hidden = None
        pred_frames, hidden_states = [], []
        for i in range(hyperparams.n_frames_out):
            x = inputs[:, -hyperparams.n_frames_in:]
            y = Y[:, i].unsqueeze(1)
            y_hat, y_hidden, loss = self(x, y, X_hidden=y_hidden)  # (B, 1, W, H)
            inputs = torch.cat([X[:, -1].unsqueeze(1), y_hat], dim=1)
            pred_frames.append(y_hat.detach().cpu())
            hidden_states.append(y_hidden.detach().cpu())
        return (pred_frames, hidden_states), loss
