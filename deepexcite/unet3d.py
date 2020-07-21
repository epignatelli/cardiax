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
from utils import log, grad_mse_loss, time_grad_loss, Elu, Downsample, Normalise, Rotate, Flip
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=2, attention=None):
        super().__init__()
        self.features = ConvBlock3D(in_channels, out_channels, kernel_size=3, padding=padding, pooling=0)
        self.upsample = nn.Upsample(scale_factor=(1, pooling, pooling))
        if attention is None:
            self.attention = nn.Identity()
        elif "self" in attention :
            self.attention = SelfAttention3D(in_channels)
        elif "soft" in attention:
            self.attention = SoftAttention3D(in_channels)

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
        attention = torch.softmax(attention, dim=1)
        attention = torch.matmul(attention, value)
        return attention


class Inference(LightningModule):
    def __init__(self, input_size, hidden_size=512):
        self.features = nn.Sequential([ConvBlock3D(8, 16, 6, 2), ConvBlock3D(16, 32, 6, 2)])
        self.mu = nn.Linear(input_size, hidden_size)
        self.logvar = nn.Linear(input_size, hidden_size)
        
    def reparameterise(self, mu, logvar):	
        std = logvar.mul(0.5).exp_()
        esp = torch.randn_like(mu, device=mu.device)
        z = mu + std * esp
        return z
    
    def get_loss(self, mu, logvar):
        kld = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kld
    
    def forward(self, x):
        x = self.features(x)
        log("inference features", x.shape)
        latent = self.flatten(x)
        mu = self.mu(latent)
        logvar = self.logvar(latent)
        z = self.reparameterise(mu, logvar)
        return z
        
    
class Unet3D(LightningModule):
    def __init__(self, channels, frames_in, frames_out, step, scale_factor=4, loss_weights={}, attention="self"):
        super().__init__()
        self.save_hyperparameters()
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.step = step
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
        recon = torch.sqrt(nn.functional.mse_loss(y_hat, y, reduction="sum") / y_hat.size(0) / y_hat.size(1))
        recon = recon * self.loss_weights.get("recon", 1.)
        space_grad = torch.sqrt(grad_mse_loss(y_hat, y, reduction="sum") / y_hat.size(0) / y_hat.size(1))
        space_grad = space_grad * self.loss_weights.get("space_grad", 1.)
        time_grad = time_grad_loss(y_hat, y) / y_hat.size(0) / y_hat.size(1)
        time_grad = time_grad * self.loss_weights.get("time_grad", 1.)
        energy = total_energy_loss(y_hat, y) / y_hat.size(0) / y_hat.size(1)
        energy = energy * self.loss_weights.get("energy", 1.)
        return {"recon": recon, "space_grad": space_grad, "time_grad": time_grad, "energy": energy}
    
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
        nrow, normalise = 10, True
        self.logger.experiment.add_image("w_input", mg(x[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("v_input", mg(x[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("u_input", mg(x[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)        
        self.logger.experiment.add_image("w_pred", mg(y_hat[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("w_truth", mg(y[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("v_pred", mg(y_hat[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("v_truth", mg(y[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("u_pred", mg(y_hat[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("u_truth", mg(y[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        return {"loss": sum(loss.values()), "log": tensorboard_logs}

        
def collate(batch):
    x = torch.stack([torch.as_tensor(t) for t in batch], 0)
    
#     # random rotate
    if random.random() > 0.5:
        x = torch.rot90(x, k=random.randint(1, 3), dims=(-2, -1))

#     # random flip
    if random.random() > 0.5:
        x = torch.flip(x, dims=(random.randint(-2, -1), ))
    return x


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--frames_in', required=True, type=int)
    parser.add_argument('--frames_out', required=True, type=int)
    parser.add_argument('--step', required=True, type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--channels', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--attention', type=str, default=None)
    
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--root', type=str, default="/media/ep119/DATADRIVE3/epignatelli/deepexcite/train_dev_set/")
    parser.add_argument('--filename', type=str, default="/media/SSD1/epignatelli/train_dev_set/spiral_params5.hdf5")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--log_interval', type=int, default=10)
    
    parser.add_argument('--recon', type=float, default=1.)
    parser.add_argument('--space_grad', type=float, default=1.)
    parser.add_argument('--time_grad', type=float, default=1.)
    parser.add_argument('--energy', type=float, default=1.)
    
    
    args = parser.parse_args()    
    utils.DEBUG = args.debug
    
    model = Unet3D(args.channels, args.frames_in, args.frames_out, args.step, scale_factor=args.scale_factor,
                   loss_weights={"recon": args.recon, "space_grad": args.space_grad, "energy": args.energy, "time_grad": args.time_grad}, attention=args.attention)
    log(model)
    log("parameters: {}".format(model.parameters_count()))
    
    fkset = FkDataset(args.root, args.frames_in, args.frames_out, args.step, transform=Normalise(), squeeze=True, keys=["spiral_params3.hdf5", "heartbeat_params3.hdf5", "three_points_params3.hdf5"])
#     fkset = Simulation(args.filename, args.frames_in, args.frames_out, args.step, transform=Normalise())
    loader = DataLoader(fkset, batch_size=args.batch_size, collate_fn=collate, shuffle=True, num_workers=3)
    trainer = Trainer.from_argparse_args(parser, fast_dev_run=args.debug, row_log_interval=args.log_interval, default_root_dir="lightning_logs/unet3d")
    trainer.fit(model, train_dataloader=loader)
    