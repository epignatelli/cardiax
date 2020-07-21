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
from utils import log, grad_mse_loss, Elu, Downsample, Normalise, Flatten, Unflatten
import random
# from sklearn.cluster import MiniBatchKMeans
import numpy as np


class KMeans(nn.Module):
    def __init__(self, n_clusters=10, n_iter=100):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.labels_ = None
        self.cluster_centers_ = None
        
    def fit(self, x):
        N, D = x.shape  # Number of samples, dimension of the ambient space

        c = x[:self.n_clusters, :].clone()  # Simplistic random initialization
        x_i = torch.tensor(x[:, None, :])  # (Npoints, 1, D)

        for i in range(self.n_iter):

            c_j = torch.tensor(c[None, :, :])  # (1, Nclusters, D)
            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
            cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

            Ncl = torch.bincount(cl)  # Class weights
            for d in range(D):  # Compute the cluster centroids with torch.bincount:
                c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl
        self.labels_ = cl
        self.cluster_centers_ = c
        

def blobify(x):
    cl = KMeans(10)
    cl.fit(x.flatten().reshape(-1, 1))

    labels = cl.labels_.reshape(x.shape)
    clusters = {}
    masks = torch.zeros((cl.n_clusters, *x.shape))
    for i in range(len(x)):
        for j in range(x.shape[1]):
            masks[labels[i, j], i, j] = 1
    return masks
    
def blobify_loss(y_hat, y):
    masks_y_hat = torch.stack([blobify(pred[2]) for pred in y_hat])
    masks_y = torch.stack([blobify(truth[2]) for truth in y])
    return nn.functional.mse_loss(y_hat, y, reduction="sum") / y_hat.size(0)


def total_energy_loss(y_hat, y):
    y_hat_energy = y_hat.sum(dim=(-3, -2, -1))
    y_energy = y.sum(dim=(-3, -2, -1))
    energy_diff = torch.abs(y_hat_energy - y_energy)
    return energy_diff.mean() / y_hat.size(0)
        

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super().__init__()
        self.features = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=padding),
                Elu(),
                nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=padding),
                Elu()
            ])
        self.downsample = nn.AvgPool2d((pooling, pooling)) if pooling else nn.Identity()

    def forward(self, x):
        log("going down")
        for i in range(len(self.features)):
            x = self.features[i](x)
            log("conv", i, x.shape)
        x = self.downsample(x)
        log("pool", i, x.shape)
        return x


class ConvTransposeBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=2, attention=True):
        super().__init__()
        self.features = ConvBlock2D(in_channels, out_channels, kernel_size=3, padding=padding, pooling=0)
        self.upsample = nn.Upsample(scale_factor=(pooling, pooling))
        self.attention = SoftAttention2D(in_channels) if attention else nn.Identity()

    def forward(self, x):
        log("going up")
        log("attention", x.shape)
        x = self.attention(x)
        log("cat", x.shape)
        x = self.features(x)
        log("conv", x.shape)
        x = self.upsample(x)
        log("upsample", x.shape)
        return x


class SoftAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.project = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.project(x))


class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention = torch.matmul(torch.transpose(query, -2, -1), key)
        attention = torch.softmax(attention, dim=-3)
        attention = torch.matmul(attention, value)
        return attention
    
class Autoencoder(LightningModule):
    def __init__(self, channels, hidden_size=512, scale_factor=2, loss_weights={}, attention=False):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = 512
        self.frames_in = 3
        self.frames_out = 3
        self.loss_weights = loss_weights
        
        down_channels = [self.frames_in] + channels if self.frames_in is not None else channels
        self.downscale = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            self.downscale.append(ConvBlock2D(down_channels[i], down_channels[i + 1], pooling=scale_factor))
        
        hidden_size = 512
        self.flatten = Flatten()
        self.latent = nn.Linear(hidden_size, hidden_size)
        self.latent = nn.Linear(hidden_size, hidden_size)
        self.unflatten = Unflatten(size=hidden_size)
        
        up_channels = [self.frames_out] + channels if self.frames_out is not None else channels
        self.upscale = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            self.upscale.append(ConvTransposeBlock2D(up_channels[-i - 1], up_channels[-i - 2], pooling=scale_factor, attention=attention))
        return
    
    def encode(self, x):
        for i in range(len((self.downscale))):
            x = self.downscale[i](x)
        return x
    
    def propagate(self, x):
        log("before flatten", x.shape)
        x = self.flatten(x)
        log("after flatten", x.shape)
        x = self.latent(x)
        log("after latent", x.shape)
        x = self.unflatten(x)
        log("after unflatten", x.shape)
        return x

    def decode(self, x):
        for i in range(len((self.upscale))):
            x = self.upscale[i](x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.propagate(x)
        x = self.decode(x)
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
#         blob_loss = blobify_loss(y_hat, y)
#         blob_loss = blob_loss * self.loss_weights.get("blob", 1.)
        return {"rmse": rmse, "rmse_grad": rmse_grad, "energy": energy}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        x = batch.float()
        log(x.shape)
        
        y_hat = self(x)
        loss = self.get_loss(y_hat, x)
        tensorboard_logs = loss
        
        i = random.randint(0, y_hat.size(0) - 1)
        self.logger.experiment.add_image("w_pred", mg(y_hat[i, 0], nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("w_truth", mg(x[i, 0], nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("v_pred", mg(y_hat[i, 1], nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("v_truth", mg(x[i, 1], nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("u_pred", mg(y_hat[i, 2], nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("u_truth", mg(x[i, 2], nrow=5, normalize=True), self.current_epoch)
        return {"loss": sum(loss.values()), "log": tensorboard_logs}

        
def collate(batch):
    return torch.stack([torch.as_tensor(t) for t in batch], 0)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--root', type=str, default="/media/ep119/DATADRIVE3/epignatelli/deepexcite/train_dev_set/")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--channels', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--attention', default=False, action="store_true")
    parser.add_argument('--rmse', type=float, default=1.)
    parser.add_argument('--rmse_grad', type=float, default=1.)
    parser.add_argument('--energy_weight', type=float, default=1.)
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--log_interval', type=int, default=10)
    
    args = parser.parse_args()    
    utils.DEBUG = args.debug
    
    model = Autoencoder(args.channels, scale_factor=args.scale_factor,
                   loss_weights={"rmse": args.rmse, "rmse_grad": args.rmse_grad, "energy": args.energy_weight}, attention=args.attention)
    log(model)
    log("parameters: {}".format(model.parameters_count()))
        
    fkset = FkDataset(args.root, 1, 0, 1, transform=Normalise(), squeeze=True, keys=["spiral_params3.hdf5", "heartbeat_params3.hdf5", "three_points_params3.hdf5"])
    loader = DataLoader(fkset, batch_size=args.batch_size, collate_fn=collate, shuffle=True, num_workers=3)
    trainer = Trainer.from_argparse_args(parser, fast_dev_run=args.debug, row_log_interval=args.log_interval, default_root_dir="lightning_logs/autoencoder")
    trainer.fit(model, train_dataloader=loader)
    