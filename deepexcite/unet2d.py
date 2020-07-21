import torch
from torch import nn
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from dataset import FkDataset, Simulation
import torchvision.transforms as t
from torch.utils.data import DataLoader
from torchvision.utils import make_grid as mg

def log(*m):
    if DEBUG:
        print(*m)

        
def grad_mse_loss(gen_frames, gt_frames, reduction="sum"):
    def gradient(x):
        log("grad", x.shape)
        left = x
        right = torch.nn.functional.pad(x, [0, 1, 0, 0])[..., :, 1:]
        top = x
        bottom = torch.nn.functional.pad(x, [0, 0, 0, 1])[..., 1:, :]

        dx, dy = right - left, bottom - top
        dx[..., :, -1] = 0
        dy[..., -1, :] = 0

        log(dx.shape, dy.shape)
#         return torch.sqrt(dx.pow(2) + dy.pow(2))
        return torch.abs(dx) + torch.abs(dy)

    grad_pred = gradient(gen_frames)
    grad_truth = gradient(gt_frames)

    # condense into one tensor and avg
    return torch.nn.functional.mse_loss(grad_pred, grad_truth, reduction=reduction)


class Elu(nn.Module):
    def forward(self, x):
        return torch.nn.functional.elu(x)


class Downsample:
    def __init__(self, size, mode="bicubic"):
        self.size = size
        self.mode = mode
    
    def __call__(self, x):
        return torch.nn.functional.interpolate(x, self.size)


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pooling=2):
        super(ConvBlock2D, self).__init__()

        self.features = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(padding)),
                Elu(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=int(padding)),
                Elu()
            ])
        self.downsample = nn.MaxPool2d((pooling, pooling)) if pooling else nn.Identity()

    def forward(self, x):
        log("going down")
        for i in range(len(self.features)):
            x = self.features[i](x)
            log("conv", i, x.shape)
        x = self.downsample(x)
        log("pool", i, x.shape)
        return x


class ConvTransposeBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super(ConvTransposeBlock2D, self).__init__()
    
        self.upsample = nn.Upsample(scale_factor=pooling)
        self.features = ConvBlock2D(in_channels, out_channels, kernel_size=3, padding=padding, pooling=0)
        
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, skip_connection):
        log("going up")
        x = self.upsample(x)
        log("upsample", x.shape)
        crop = self.center_crop(skip_connection, x.shape[2:])
        x = torch.cat([x, crop], 1)
        log("cat", x.shape)
        x = self.features(x)
        log("conv", x.shape)
        return out

    
class Autoencoder(LightningModule):
    def __init__(self, channels, first_channel, last_channel, scale_factor=4):
        super(Autoencoder, self).__init__()
        self.frames_in = first_channel
        self.frames_out = last_channel
        
        # downsampling
        self.downsample = nn.ModuleList()
        in_channels = self.frames_in
        for i in range(len(channels)):
            self.downsample.append(ConvBlock2D(in_channels=in_channels, out_channels=channels[i], pooling=scale_factor))
            in_channels = channels[i]

        # upsample
        self.upsample = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.upsample.append(ConvTransposeBlock2D(in_channels=in_channels, out_channels=channels[i], pooling=scale_factor))
            in_channels = channels[i]

        # return only one frame out
        self.output = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)


    def forward(self, X, Y=None):
        skip_connections = []
        for i, down in enumerate(self.downsample):
            X = down(X)
            if i != len(self.downsample) - 1:
                skip_connections.append(X)

        for i, up in enumerate(self.upsample):
            X = up(X, skip_connections[-i - 1])

        X = self.output(X)
        
    def get_loss(self, y_hat, y):
        rmse = torch.sqrt(nn.functional.mse_loss(y_hat, y, reduction="sum") / y_hat.size(0))
        rmse_grad = torch.sqrt(grad_mse_loss(y_hat, y, reduction="sum") / y_hat.size(0))
        return rmse + (0.01 * rmse_grad).exp()
    
    def encode(self, X):
        skip_connections = []
        z = X
        for i, down in enumerate(self.downsample):
            z = down(z)
            if i != len(self.downsample) - 1:
                skip_connections.append(z)
        return z, skip_connections
    
    def decode(self, x, skip_connections):
        for i, up in enumerate(self.upsample):
            x = up(x, skip_connections[-i - 1])
        x = self.output(x)
        return y_hat
    
    def propagate(self, z):
        return z
    
    def forward(self, X):
        x, connections = self.encode(X)
        log("skip connections", [x.shape for x in connections])
        z = self.propagate(x)
        y_hat = self.decode(z, connections)
        return y_hat
    
    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_loss(self, y_hat, y):
        rmse = torch.sqrt(nn.functional.mse_loss(y_hat, y, reduction="sum") / y_hat.size(0))
        rmse_grad = torch.sqrt(grad_mse_loss(y_hat, y, reduction="sum") / y_hat.size(0))
        return rmse + (0.01 * rmse_grad.exp())
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    
    def training_step(self, batch, batch_idx):
        batch = batch.float()
        x = batch[:, :self.frames_in]
        y = batch[:, self.frames_in:]
        log(x.shape)
        log(y.shape)
        
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        
        self.logger.experiment.add_image("u_pred", mg(y_hat[0].unsqueeze(1), normalize=True), self.current_epoch)
        self.logger.experiment.add_image("u_truth", mg(y[0].unsqueeze(1), normalize=True), self.current_epoch)
        return {"loss": loss, "log": tensorboard_logs}

        
def collate(batch):
    return torch.stack([torch.as_tensor(t[:, 2]) for t in batch], 0)


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--root', type=str, default="/media/SSD1/epignatelli/train_dev_set")
    parser.add_argument('--filename', type=str, default="/media/SSD1/epignatelli/train_dev_set/spiral_params5.hdf5")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--frames_in', type=int, default=5)
    parser.add_argument('--frames_out', type=int, default=10)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--channels', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--gpus', type=str, default="0")
    
    args = parser.parse_args()    
    DEBUG = args.debug
    
    model = UNet2D(args.channels, args.frames_in, args.frames_out, scale_factor=args.scale_factor)
    log(model)
    log("parameters: {}".format(model.parameters_count()))
        
#     fkset = FkDataset(ROOT, FRAMES_IN, FRAMES_OUT, 1, transforms=t.Compose([Downsample((INPUT_SIZE, INPUT_SIZE))]), squeeze=True)
    fkset = Simulation(args.filename, args.frames_in, args.frames_out, args.step)
    loader = DataLoader(fkset, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
    trainer = Trainer.from_argparse_args(parser, fast_dev_run=args.debug)
    trainer.fit(model, train_dataloader=loader)