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
from utils import log, scream
from utils import Elu
from utils import time_grad, space_grad_mse_loss, time_grad_mse_loss, energy_mse_loss
from utils import Downsample, Normalise, Rotate, Flip, Noise
import random
import math
from functools import partial
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks.base import Callback


class IncreaseFramsesOut(Callback):
    def __init__(self, monitor="loss", trigger_at=1.6e-3, max_value=20):
        self.monitor = monitor
        self.trigger_at = trigger_at
        self.max_value = max_value
        
    def on_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get(self.monitor)
        if loss is None:
            print("WARNING: IncreaseFramesOut callback failed. Cannot retrieve metric {}".format(self.monitor))
            return
        if loss <= self.trigger_at:
            if trainer.model.frames_out >= self.max_value:
                print("Epoch\t{}: hit max number of output frames {}".format(trainer.current_epoch, trainer.model.frames_out))
                return
            trainer.model.frames_out += 1
            trainer.train_dataloader.dataset.frames_out += 1
            trainer.val_dataloaders[0].dataset.frames_out += 1
            print("Epoch\t{}: increasing number of output frames at {}".format(trainer.current_epoch, trainer.model.frames_out))
        return

    
class SplineActivation(nn.Module):
    def __init__(self, degree):
        super(SplineActivation, self).__init__()
        self.weights = nn.Parameter(torch.empty(degree + 1))
    
    def forward(self, x):
        for i in range(len(self.weights)):
            x = x + (x ** i) * self.weights[i]
        return x


class SoftAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.project = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x):
        return torch.sigmoid(self.project(x))


class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention = torch.matmul(torch.transpose(query, -2, -1), key)
        attention = torch.softmax(attention, dim=1)
        attention = torch.matmul(attention, value)
        return attention

    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, attention="none", activation=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if activation:
            self.activation = SplineActivation(activation)
        else:
            self.activation = torch.nn.functional.elu
        if attention is None or attention.lower() == "none":
            self.attention = None
        elif "self" in attention:
            self.attention = SelfAttention3D(out_channels)
        else:
            self.attention = SoftAttention3D(out_channels)
        
    def forward(self, x):
        log("x: ", x.shape)
        dx = self.conv(x)
        dx = self.activation(dx)
        log("conv: ", dx.shape)
        if self.attention is not None:
            a = self.attention(x)
            log("attention: ", a.shape)
            return (dx * a) + x
        return dx + x
    
    
class ResNet(LightningModule):
    def __init__(self, n_layers, n_filters, kernel_size, residual_step, activation, frames_in, frames_out, step, loss_weights={}, attention="self", lr=0.001, profile=False):
        super().__init__()
        # public:
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.residual_step  = residual_step
        self.activation = activation
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.step = step
        self.loss_weights = loss_weights
        self.attention = attention
        self.lr = lr
        self.profile = profile
        
        # private:
        self.register_buffer("_log_gpu_mem_step", torch.tensor(0.))
        
        padding = tuple([math.floor(x / 2) for x in kernel_size])
        self.inlet = nn.Conv3d(self.frames_in, n_filters, kernel_size=kernel_size, stride=1, padding=padding)
        
        self.flow = nn.ModuleList()
        for i in range(n_layers):
            self.flow.append(ResidualBlock(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=padding,
                                           attention=attention, activation=activation))
            
        self.outlet = nn.Conv3d(n_filters, 1, kernel_size=kernel_size, stride=1, padding=padding)
        
    def forward(self, x):
        x = self.inlet(x)
        queue = []
        for i, m in enumerate(self.flow):
            x = m(x)
            if not (i % self.residual_step):
                queue.append(x)
                if i != self.residual_step:
                    res = queue.pop()
                    x = x + res
        x = self.outlet(x)
        return x
    
    def backward(self, *args):
        # we're using truncated backprop through time where the inputs depend on new outputs
        # the backward methods is called inside training_step
        # we override this to do nothing
        return
    
    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_loss(self, y_hat, y):
        recon_loss = torch.sqrt(nn.functional.mse_loss(y_hat, y, reduction="mean")) / y_hat.size(0) / y_hat.size(1)
        recon_loss = recon_loss * self.loss_weights.get("recon_loss", 1.)
        space_grad_loss = torch.sqrt(space_grad_mse_loss(y_hat, y, reduction="mean")) / y_hat.size(0) / y_hat.size(1)
        space_grad_loss = space_grad_loss * self.loss_weights.get("space_grad_loss", 1.)
        energy_loss = torch.sqrt(energy_mse_loss(y_hat, y, reduction="mean")) / y_hat.size(0) / y_hat.size(1)
        energy_loss = energy_loss * self.loss_weights.get("energy_loss", 1.)
        return {"recon_loss": recon_loss, "space_grad_loss": space_grad_loss, "energy_loss": energy_loss}
    
    def configure_optimizers(self):
        optimisers = [torch.optim.Adam(self.parameters(), lr=self.lr)]
        schedulers = [{
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimisers[0], verbose=True, min_lr=1e-6),
            'monitor': 'loss'}]
        return optimisers, schedulers

    def profile_gpu_memory(self):
        if not self.profile:
            return
        memory = torch.cuda.memory_allocated(self.inlet.bias.device) / 1024 / 1024
        self.logger.experiment.add_scalar("gpu_mem", memory, self._log_gpu_mem_step)
        log("gpu_memory", memory)
        self._log_gpu_mem_step += 1
        return

    def on_train_start(self):
        # add graph with a randomly-sized input
        self.logger.experiment.add_graph(self, torch.randn((16, self.frames_in, 3, 256, 256), device=self.inlet.bias.device))        
        return
        
    def training_step(self, batch, batch_idx):
        x = batch[:, :self.frames_in]
        y = batch[:, self.frames_in:]
        self.profile_gpu_memory()
        
        output_sequence = torch.empty_like(y, requires_grad=False, device=self.device).type_as(x)
        loss = {}
        for i in range(self.frames_out):
            # forward pass
            y_hat = self(x).squeeze()
            self.profile_gpu_memory()
            
            # calculate loss
            current_loss = self.get_loss(y_hat, y[:, i])
            y_hat = y_hat.detach()
            total_loss = sum(current_loss.values())
            for k, v in current_loss.items():
                loss.update({k: (loss.get(k, 0.) + v)})
            self.profile_gpu_memory()
            
            # backward pass
            total_loss.backward()
            self.profile_gpu_memory()
            
            # update output sequence
            output_sequence[:, i] = y_hat
            self.profile_gpu_memory()

            # update input sequence with predicted frames
            if (self.frames_out > 1):
                x = torch.cat([x[:, -(self.frames_in - 1):], y_hat.unsqueeze(1)], dim=1)
                self.profile_gpu_memory()
            
        # logging losses
        logs = {"train_loss/" + k: v for k, v in loss.items()}
        logs["train_loss/total_loss"] = total_loss
        return {"loss": total_loss, "log": logs, "x": batch[:, :self.frames_in], "y_hat": output_sequence, "y": y}
    
    def training_epoch_end(self, outputs):            
        # log outputs as images
        x = random.choice([x['x'] for x in outputs])
        y_hat = random.choice([x['y_hat'] for x in outputs])
        y = random.choice([x['y'] for x in outputs])
        i = random.randint(0, y_hat.size(0) - 1)
        nrow, normalise = 10, True
        self.logger.experiment.add_image("train_w/input", mg(x[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("train_v/input", mg(x[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("train_u/input", mg(x[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("train_w/pred", mg(y_hat[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("train_v/pred", mg(y_hat[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("train_u/pred", mg(y_hat[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("train_w/truth", mg(y[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("train_v/truth", mg(y[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("train_u/truth", mg(y[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        
        # average loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {"loss": avg_loss}
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch[:, :self.frames_in]
        y = batch[:, self.frames_in:]
        self.profile_gpu_memory()
        
        output_sequence = torch.empty_like(y, requires_grad=False, device=self.device).type_as(x)
        loss = {}
        for i in range(self.frames_out):  # 10
            # forward pass
            y_hat = self(x).squeeze()
            self.profile_gpu_memory()
            
            # calculate loss
            current_loss = self.get_loss(y_hat, y[:, i])
            y_hat = y_hat.detach()
            total_loss = sum(current_loss.values())
            for k, v in current_loss.items():
                loss.update({k: (loss.get(k, 0.) + v)})  # detach partial losses since they're not useful anymore for backprop
            self.profile_gpu_memory()
            
            # update output sequence
            output_sequence[:, i] = y_hat
            self.profile_gpu_memory()

            # update input sequence with predicted frames
            if (self.frames_out > 1):
                x = torch.cat([x[:, -(self.frames_in - 1):], y_hat.unsqueeze(1)], dim=1)
                self.profile_gpu_memory()
            
        # logging losses
        logs = {"train_loss/" + k: v for k, v in loss.items()}
        logs["train_loss/total_loss"] = total_loss
        return {"loss": total_loss, "log": logs, "x": batch[:, :self.frames_in], "y_hat": output_sequence, "y": y}
    
    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        # log loss
        loss = {}
        for i in range(len(outputs)):
            for k, v in outputs[i]["log"].items():
                loss.update({k: (loss.get(k, 0.) + v)})
        for k, v in loss.items():
            self.logger.experiment.add_scalar(k, v / len(outputs), self.current_epoch)
        
        # log outputs as images
        x = random.choice([x['x'] for x in outputs])
        y_hat = random.choice([x['y_hat'] for x in outputs])
        y = random.choice([x['y'] for x in outputs])
        i = random.randint(0, y_hat.size(0) - 1)
        nrow, normalise = 10, True
        self.logger.experiment.add_image("val_v/input", mg(x[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_w/input", mg(x[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_u/input", mg(x[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_v/pred", mg(y_hat[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_w/pred", mg(y_hat[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_u/pred", mg(y_hat[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_v/truth", mg(y[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_w/truth", mg(y[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_u/truth", mg(y[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        return {"loss": sum(loss.values())}

    def on_epoch_end(self):
        # log model weights
        for i, module in enumerate(self.flow):
            # conv kernels
            self.logger.experiment.add_image("kernel/layer-{}".format(i), mg(module.conv.weight[0], nrow=self.n_filters, normalize=True), self.current_epoch)
            
            # log attention 
            if hasattr(module.attention, "project"):
                self.logger.experiment.add_histogram("_soft-attention", module.attention.project.weight, i)
                self.logger.experiment.add_image("_soft-attention/layer-{}".format(i), mg(module.conv.weight[0], nrow=self.n_filters, normalize=True), self.current_epoch)
            elif hasattr(module.attention, "query"):
                self.logger.experiment.add_histogram("_seft-attention/query", module.attention.query.weight, i)
                self.logger.experiment.add_histogram("_seft-attention/key", module.attention.key.weight, i)
                self.logger.experiment.add_histogram("_seft-attention/value", module.attention.value.weight, i)
                self.logger.experiment.add_image("_seft-attention/query-{}".format(i), mg(module.attention.query.weight[0], nrow=self.n_filters, normalize=True), self.current_epoch)
                self.logger.experiment.add_image("_seft-attention/key-{}".format(i), mg(module.attention.key.weight[0], nrow=self.n_filters, normalize=True), self.current_epoch)
                self.logger.experiment.add_image("_seft-attention/value-{}".format(i), mg(module.attention.query.weight[0], nrow=self.n_filters, normalize=True), self.current_epoch)
        return

    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # model args
    parser.add_argument('--frames_in', type=int, default=2)
    parser.add_argument('--frames_out', type=int, default=1)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--n_filters', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, nargs='+', default=(1, 7, 7))
    parser.add_argument('--residual_step', type=int, default=5)
    parser.add_argument('--activation', type=int, default=0)
    parser.add_argument('--attention', type=str, default="none")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--recon_loss', type=float, default=1.)
    parser.add_argument('--space_grad_loss', type=float, default=1.)
    parser.add_argument('--time_grad_loss', type=float, default=1.)
    parser.add_argument('--energy_loss', type=float, default=0.)
    
    # loader args
    parser.add_argument('--root', type=str, default="/media/ep119/DATADRIVE3/epignatelli/deepexcite/train_dev_set/")
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=3)
    
    # trainer args
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--profile', default=False, action="store_true")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--distributed_backend', type=str, default=None)
    parser.add_argument('--row_log_interval', type=int, default=10)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    
    
    args = parser.parse_args()
    utils.DEBUG = DEBUG = args.debug
    
    # define loss weights
    loss_weights = {
        "recon_loss": args.recon_loss,
        "space_grad_loss": args.space_grad_loss,
        "energy_loss": args.energy_loss,
        "time_grad_loss": args.time_grad_loss
    }
    
    # define model
    model = ResNet(n_layers=args.n_layers,
                   n_filters=args.n_filters,
                   kernel_size=tuple(args.kernel_size),
                   residual_step=args.residual_step,
                   activation=args.activation,
                   frames_in=args.frames_in, frames_out=args.frames_out, step=args.step,
                   loss_weights=loss_weights,
                   attention=args.attention,
                   lr=args.lr,
                   profile=args.profile)
    
    # print debug info
    log(model)
    log("parameters: {}".format(model.parameters_count()))

    # train_dataloader
    train_transform = t.Compose([torch.as_tensor, Normalise(), Rotate(), Flip(), Noise(args.frames_in)])
    train_fkset = FkDataset(args.root, args.frames_in, args.frames_out, args.step, transform=train_transform, squeeze=True, keys=["spiral_params3.hdf5", "three_points_params3.hdf5"])
    train_loader = DataLoader(train_fkset, batch_size=args.batch_size, collate_fn=torch.stack, shuffle=True, num_workers=args.n_workers, drop_last=True, pin_memory=True)
    
    # val_dataloader
    val_transform = t.Compose([torch.as_tensor, Normalise()])
    val_fkset = FkDataset(args.root, args.frames_in, args.frames_out, args.step, transform=val_transform, squeeze=True, keys=["heartbeat_params3.hdf5"])
    val_loader = DataLoader(val_fkset, batch_size=args.batch_size, collate_fn=torch.stack, num_workers=args.n_workers, drop_last=True, pin_memory=True)

    # begin training
    trainer = Trainer.from_argparse_args(parser,
                                         fast_dev_run=args.debug,
                                         default_root_dir="lightning_logs/resnet",
                                         profiler=args.profile,
                                         log_gpu_memory="all" if args.profile else None,
                                         train_percent_check=0.01 if args.profile else 1.0,
                                         val_percent_check=0.1 if args.profile else 1.0,
                                         callbacks=[LearningRateLogger(), IncreaseFramsesOut(trigger_at=1.6e-3 if not args.profile else 10.)])
    
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    