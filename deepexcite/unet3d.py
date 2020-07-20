import torch
from torch import nn
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from dataset import FkDataset, Simulation
import torchvision.transforms as t
from torch.utils.data import DataLoader
from torchvision.utils import make_grid as mg
import utils
from utils import log, grad_mse_loss, Elu, Downsample, Normalise
import random


def total_energy_loss(y_hat, y):
    y_hat_energy = y_hat.sum(dim=(2, 3, 4))
    y_energy = y.sum(dim=(2, 3, 4))
    energy_diff = torch.abs(y_hat_energy - y_energy)
    return energy_diff.mean() / y_hat.size(0) / y_hat.size(1)
        

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super().__init__()
        self.features = nn.ModuleList(
            [
                nn.Conv3d(in_channels, out_channels, kernel_size=(3, kernel_size, kernel_size), padding=padding),
                Elu(),
                nn.Conv3d(out_channels, out_channels, kernel_size=(3, kernel_size, kernel_size), padding=padding),
                Elu()
            ])
        self.downsample = nn.AvgPool3d((1, pooling, pooling)) if pooling else nn.Identity()

    def forward(self, x):
        log("going down")
        for i in range(len(self.features)):
            x = self.features[i](x)
            log("conv", i, x.shape)
        x = self.downsample(x)
        log("pool", i, x.shape)
        return x


class ConvTransposeBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=2, attention=True):
        super().__init__()
        self.features = ConvBlock3D(in_channels, out_channels, kernel_size=3, padding=padding, pooling=0)
        self.upsample = nn.Upsample(scale_factor=(1, pooling, pooling))
        self.attention = SoftAttention3D(in_channels) if attention else nn.Identity()

    def forward(self, x, skip_connection):
        log("going up")
        log("attention", x.shape)
        skip_connection = self.attention(skip_connection)
        log("cat", x.shape)
        x = self.features(x + skip_connection)
        log("conv", x.shape)
        x = self.upsample(x)
        log("upsample", x.shape, skip_connection.shape)
        return x


class SoftAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.project = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1))

    def forward(self, x):
        return torch.sigmoid(self.project(x))


class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1))
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1))
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1))
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention = torch.matmul(torch.transpose(query, -2, -1), key)
        attention = torch.softmax(attention, dim=-3)
        attention = torch.matmul(attention, value)
        return attention
    
class Unet3D(LightningModule):
    def __init__(self, channels, frames_in, frames_out, scale_factor=4, loss_weights={}, attention=False):
        super().__init__()
        self.save_hyperparameters()
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.loss_weights = loss_weights
        
        down_channels = [frames_in] + channels if frames_in is not None else channels
        self.downscale = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            self.downscale.append(ConvBlock3D(down_channels[i], down_channels[i + 1], pooling=scale_factor))
        
        up_channels = [frames_out] + channels if frames_out is not None else channels
        self.upscale = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            self.upscale.append(ConvTransposeBlock3D(up_channels[-i - 1], up_channels[-i - 2], pooling=scale_factor, attention=attention))
        return
    
    def encode(self, x):
        skip_connections = []
        for i in range(len((self.downscale))):
            x = self.downscale[i](x)
            skip_connections.append(x)
        return x, skip_connections
    
    def propagate(self, x):
        # implement normalizing flow here
        return x
    
    def decode(self, x, skip_connections):
        for i in range(len((self.upscale))):
            x = self.upscale[i](x, skip_connections[-i - 1])
        return x
    
    def forward(self, x):
        x, skip_connections = self.encode(x)
        log("skip connections", [x.shape for x in skip_connections])
        x = self.propagate(x)
        x = self.decode(x, skip_connections)
        return x
    
    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_loss(self, y_hat, y):
        rmse = torch.sqrt(nn.functional.mse_loss(y_hat, y, reduction="sum") / y_hat.size(0) / y_hat.size(1))
        rmse_grad = torch.sqrt(grad_mse_loss(y_hat, y, reduction="sum") / y_hat.size(0) / y_hat.size(1))
        rmse = rmse * self.loss_weights.get("rmse", 1.)
        rmse_grad = rmse_grad * self.loss_weights.get("rmse_grad", 1.)
        energy = total_energy_loss(y_hat, y)
        energy = energy * self.loss_weights.get("energy", 1.)
        return {"rmse": rmse, "rmse_grad": rmse_grad, "energy": energy}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        batch = batch.float()
        x = batch[:, :self.frames_in]
        y = batch[:, self.frames_in:]
        log(x.shape)
        log(y.shape)
        
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        tensorboard_logs = loss
        
        i = random.randint(0, y_hat.size(0) - 1)
        self.logger.experiment.add_image("w_pred", mg(y_hat[i, :, 0].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("w_truth", mg(y[i, :, 0].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("v_pred", mg(y_hat[i, :, 1].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("v_truth", mg(y[i, :, 1].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("u_pred", mg(y_hat[i, :, 2].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("u_truth", mg(y[i, :, 2].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        return {"loss": sum(loss.values()), "log": tensorboard_logs}

        
def collate(batch):
    return torch.stack([torch.as_tensor(t) for t in batch], 0)


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--root', type=str, default="/media/ep119/DATADRIVE3/epignatelli/deepexcite/train_dev_set/")
    parser.add_argument('--filename', type=str, default="/media/SSD1/epignatelli/train_dev_set/spiral_params5.hdf5")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--frames_in', type=int, default=5)
    parser.add_argument('--frames_out', type=int, default=10)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--channels', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--attention', default=False, action="store_true")
    parser.add_argument('--rmse', type=float, default=1.)
    parser.add_argument('--rmse_grad', type=float, default=1.)
    parser.add_argument('--energy_weight', type=float, default=1.)
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--log_interval', type=int, default=10)
    
    args = parser.parse_args()    
    utils.DEBUG = args.debug
    
    model = Unet3D(args.channels, args.frames_in, args.frames_out, scale_factor=args.scale_factor,
                   loss_weights={"rmse": args.rmse, "rmse_grad": args.rmse_grad, "energy": args.energy_weight}, attention=args.attention)
    log(model)
    log("parameters: {}".format(model.parameters_count()))
        
    fkset = FkDataset(args.root, args.frames_in, args.frames_out, args.step, squeeze=True, keys=["spiral_params3.hdf5", "heartbeat_params3.hdf5", "three_points_params3.hdf5"])
#     fkset = Simulation(args.filename, args.frames_in, args.frames_out, args.step, transform=Normalise())
    loader = DataLoader(fkset, batch_size=args.batch_size, collate_fn=collate, shuffle=True, num_workers=3)
    trainer = Trainer.from_argparse_args(parser, fast_dev_run=args.debug, row_log_interval=args.log_interval)
    trainer.fit(model, train_dataloader=loader)
    