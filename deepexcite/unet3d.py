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
        self.downsample = nn.MaxPool3d((1, pooling, pooling)) if pooling else nn.Identity()

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
        self.upsample = nn.Upsample(scale_factor=(1, pooling, pooling))
        self.features = ConvBlock3D(in_channels, out_channels, kernel_size=3, padding=padding, pooling=0)
#         self.attention = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, skip_connection):
        log("going up")
        log("attention", skip_connection.shape)
#         skip_connection = self.attention(skip_connection)
        log("cat", x.shape)
        x = self.features(x + skip_connection)
        log("conv", x.shape)
        x = self.upsample(x)
        log("upsample", x.shape, skip_connection.shape)      
        return x


class Unet3D(LightningModule):
    def __init__(self, channels, frames_in, frames_out, scale_factor=4, loss_weights={}):
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
            self.upscale.append(ConvTransposeBlock3D(up_channels[-i - 1], up_channels[-i - 2], pooling=scale_factor))
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
        rmse = torch.sqrt(nn.functional.mse_loss(y_hat, y, reduction="sum") / y_hat.size(0))
        rmse_grad = torch.sqrt(grad_mse_loss(y_hat, y, reduction="sum") / y_hat.size(0))
        rmse = rmse * self.loss_weights.get("rmse", 1.)
        rmse_grad = rmse_grad * self.loss_weights.get("rmse_grad", 1.)
        return {"rmse": rmse, "rmse_grad": rmse_grad}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    
    def training_step(self, batch, batch_idx):
        batch = batch.float()
#         batch = torch.nn.functional.interpolate(batch, 512)
        x = batch[:, :self.frames_in]
        y = batch[:, self.frames_in:]
        log(x.shape)
        log(y.shape)
        
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        tensorboard_logs = loss
        
        self.logger.experiment.add_image("w_pred", mg(y_hat[0, :, 0].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("w_truth", mg(y[0, :, 0].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("v_pred", mg(y_hat[0, :, 1].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("v_truth", mg(y[0, :, 1].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("u_pred", mg(y_hat[0, :, 2].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        self.logger.experiment.add_image("u_truth", mg(y[0, :, 2].unsqueeze(1), nrow=5, normalize=True), self.current_epoch)
        return {"loss": sum(loss.values()), "log": tensorboard_logs}

        
def collate(batch):
    return torch.stack([torch.as_tensor(t) for t in batch], 0)


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
    parser.add_argument('--rmse', type=float, default=1.)
    parser.add_argument('--rmse_grad', type=float, default=1.)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=10)
    
    args = parser.parse_args()    
    DEBUG = args.debug
    
    model = Unet3D(args.channels, args.frames_in, args.frames_out, scale_factor=args.scale_factor, loss_weights={"rmse": args.rmse, "rmse_grad": args.rmse_grad})
    log(model)
    log("parameters: {}".format(model.parameters_count()))
        
#     fkset = FkDataset(ROOT, FRAMES_IN, FRAMES_OUT, 1, transforms=t.Compose([Downsample((INPUT_SIZE, INPUT_SIZE))]), squeeze=True)
    fkset = Simulation(args.filename, args.frames_in, args.frames_out, args.step)
    loader = DataLoader(fkset, batch_size=args.batch_size, collate_fn=collate, shuffle=True, num_workers=8)
    trainer = Trainer.from_argparse_args(parser, fast_dev_run=args.debug, row_log_interval=args.log_interval)
    trainer.fit(model, train_dataloader=loader)
    