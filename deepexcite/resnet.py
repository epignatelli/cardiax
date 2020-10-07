from pytorch_lightning import Trainer, seed_everything
seed_everything(2)
import logging
import math
import os
import re
import sys
from functools import partial
from typing import Tuple
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torchvision.transforms as t
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.profiler import AdvancedProfiler
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import plot
from dataset import ConcatSequence
from deepexcite.utils import (Flip, Normalise, Rotate, energy_mse_loss,
                              space_grad_mse_loss, Downsample)


class IncreaseFramsesOut(Callback):
    def __init__(self, monitor: str="checkpoint_on", at_loss: float=None, every_k_epochs: int=None, max_value: int=20):
        self.monitor = monitor
        self.at_loss = at_loss
        self.every_k_epochs = every_k_epochs
        self.max_value = max_value

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # get final loss (if ddp, the model we're in)
        loss = trainer.callback_metrics.get(self.monitor)
        if loss is None:
            logging.warning("IncreaseFramesOut callback failed. Cannot retrieve metric {}".format(self.monitor))
            return

        # synch and average if ddp
        if trainer.use_ddp or trainer.use_ddp2 or trainer.use_dp or trainer.use_tpu:
            dist.all_reduce(loss)
            loss /= abs(dist.get_world_size())
        logging.info("\nIncreaseFramsesOut callback: loss is {}".format(loss))

        if (self.at_loss is not None and loss <= self.at_loss) or (self.every_k_epochs is not None and trainer.current_epoch % self.every_k_epochs):
            if pl_module.hparams.frames_out >= self.max_value:
                logging.info("\nEpoch\t{} - IncreaseFramsesOut callback: Max number of output frames reached")
                return
            pl_module.hparams.frames_out += 1
            trainer.train_dataloader.dataset.frames_out = pl_module.hparams.frames_out
            trainer.val_dataloaders[0].dataset.frames_out = pl_module.hparams.frames_out
            assert pl_module.hparams.frames_out == trainer.train_dataloader.dataset.frames_out == trainer.val_dataloaders[0].dataset.frames_out
            logging.info("\nEpoch\t{} - IncreaseFramsesOut callback: Increase number of output frames")
        else:
            logging.info("\nEpoch\t{} - IncreaseFramsesOut callback: Do not increase the number of output frames")

        logging.info("\nIncreaseFramsesOut callback: model.frames_out: {}; train_loader.frames_out: {}; val_loader.frames_out: {}".format(
              pl_module.hparams.frames_out, trainer.train_dataloader.dataset.frames_out, trainer.val_dataloaders[0].dataset.frames_out))
        # log event
        pl_module.logger.experiment.add_scalar("frames_out", pl_module.hparams.frames_out, trainer.current_epoch + 1)
        return


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int], stride: Tuple[int], padding: Tuple[int], hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or out_channels
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv3d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.elu2 = nn.ELU()

    def forward(self, x):
        dx = self.conv1(x)
        dx = self.elu1(dx)
        dx = self.conv2(dx)
        dx = dx.add(x)
        return self.elu2(dx)


class ResNet(LightningModule):
    def __init__(self,
                 n_layers,
                 n_filters,
                 n_hidden_filters,
                 kernel_size,
                 n_latent_channels,
                 size,
                 frames_in,
                 frames_out,
                 step,
                 root,
                 train_search_regex,
                 val_search_regex,
                 batch_size,
                 n_workers,
                 loss_weights={},
                 lr=0.001,
                 profile=False,
                 time_conv=True):
        super().__init__()
        self.save_hyperparameters()

        # public:
        self.hparams.n_layers = n_layers
        self.hparams.n_filters = n_filters
        self.hparams.n_hidden_filters = n_hidden_filters
        self.hparams.kernel_size = kernel_size
        self.hparams.n_latent_channels = n_latent_channels
        self.hparams.size = size
        self.hparams.frames_in = frames_in
        self.hparams.frames_out = frames_out
        self.hparams.step = step
        self.hparams.loss_weights = loss_weights
        self.hparams.batch_size = batch_size
        self.hparams.train_search_regex = train_search_regex
        self.hparams.val_search_regex = val_search_regex
        self.lr = lr
        self.profile = profile
        self.root = root
        self.n_workers = n_workers

        # private:
        self._log_gpu_mem_step = 0

        # pad for same size output
        padding = tuple([math.floor(x / 2) for x in kernel_size])

        # diffusivity map
        p = padding[-1]
        diffusivity = torch.ones(batch_size, frames_in, 1, *size, device=self.device, requires_grad=False)
        diffusivity = torch.nn.functional.pad(diffusivity, pad=(p, p, p, p), mode="constant", value=0.)
        self.register_buffer("diffusivity", diffusivity)
        self.padding = padding
        self.input_shape = (batch_size, frames_in, 1 + 3 + n_latent_channels, size[0] + 2 * p, size[1] + 2 * p)
        self.example_input_array = torch.randn(self.input_shape)

        # init parameters
        self.inlet = nn.Conv3d(frames_in, n_filters, kernel_size=kernel_size, stride=1, padding=padding)
        self.flow = nn.Sequential(*[ResidualBlock(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=padding) for _ in range(n_layers)])
        self.outlet = nn.Conv3d(n_filters, 1, kernel_size=kernel_size, stride=1, padding=padding)

    def init_hidden(self):
        shape = (self.hparams.batch_size, self.hparams.frames_in, self.hparams.n_latent_channels, *self.input_shape[-2:])
        h = torch.zeros(shape, device=self.device)
        return h

    def forward(self, x):
        # add static diffusivity field and latent channels to
        y_hat = torch.cat([self.diffusivity, x], dim=2)
        # add latent channels
        # if h is None:
        #     h = self.init_hidden()
        # x = torch.cat([x, h], dim=2)
        # forward
        y_hat = self.inlet(y_hat)
        y_hat = self.flow(y_hat)
        y_hat = self.outlet(y_hat)
        # y_hat, h = x.split((x.size(2) - self.hparams.n_latent_channels, self.hparams.n_latent_channels), dim=2)
        # h = torch.sigmoid(h)

        # remove diffusivity channel
        y_hat = y_hat[:, :, 1:]
        return y_hat

    def backward(self, *args):
        return

    def compute_loss(self, y_hat, y):
        recon_loss = nn.functional.mse_loss(y_hat, y) / self.hparams.batch_size
        recon_loss = recon_loss.mul(self.hparams.loss_weights.get("recon_loss", 1.))
        grad_loss = space_grad_mse_loss(y_hat, y) / self.hparams.batch_size
        grad_loss = grad_loss.mul(self.hparams.loss_weights.get("grad_loss", 1.))
        total_loss = recon_loss.add(grad_loss)
        return {"recon_loss": recon_loss, "grad_loss": grad_loss, "total_loss": total_loss}

    def configure_optimizers(self):
        optimisers = [torch.optim.Adam(self.parameters(), lr=self.lr)]
        schedulers = [{
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimisers[0], verbose=True, min_lr=1e-4),
            'monitor': 'checkpoint_on'}]
        return optimisers, schedulers

    def profile_gpu_memory(self):
        if not self.profile:
            return
        memory = torch.cuda.memory_allocated(self.inlet.bias.device) / 1024 / 1024
        self.logger.experiment.add_scalar("gpu_mem", memory, self._log_gpu_mem_step)
        logging.debug("gpu_memory: {}".format(memory))
        self._log_gpu_mem_step += 1
        return

    def preprocess_input(self, batch):
        p = tuple([math.floor(x / 2) for x in self.hparams.kernel_size])[-1]
        batch = Normalise(batch)
        # if self.training:
        #     batch = Rotate(batch)
        #     batch = Flip(batch)
        batch = torch.nn.functional.pad(batch, pad=(p, p, p, p), mode="constant",  value=0.)
        return batch

    def learning_step(self, batch, batch_idx):
        batch = self.preprocess_input(batch)
        x, y = batch.split((self.hparams.frames_in, self.hparams.frames_out), dim=1)

        losses = {}
        y_hat_stacked = torch.empty_like(y, requires_grad=False, device=self.device)  # (batch_size, frames_out, 2, 256, 256)
        for i in range(self.hparams.frames_out):
            # compute loss
            y_hat = self(x)
            loss = self.compute_loss(y_hat, y[:, i:i + 1])

            # tbbtt backward
            if self.training:
               loss["total_loss"].backward()

            # accumulate losses
            for k, v in loss.items():
                losses.update({k: (losses.get(k, 0.) + v)})

            # store predictions
            y_hat_stacked[:, i] = y_hat.squeeze(1)

            # update input
            if (self.hparams.frames_out > 1):
                x = (torch.cat([x[:, -(self.hparams.frames_in - 1):], y_hat], dim=1)).detach()  # (batch_size, frames_in - 1, 1, 256, 256)

        # log losses
        if self.trainer is not None and not (self.global_step % self.trainer.row_log_interval):
            self.log_step(y_hat_stacked, y, losses)

        result = pl.TrainResult(losses["total_loss"]) if self.training else pl.EvalResult(checkpoint_on=losses["total_loss"])
        return result

    def training_step(self, batch, batch_idx):
       return self.learning_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.learning_step(batch, batch_idx)

    @torch.no_grad()
    def infer(self, x):
        if x.dim() == 4:  # no batch dimension
            x = x.unsqueeze(0)
        output_sequence = torch.empty((1, self.frames_out, 3, x.size(-1), x.size(-1)), device=torch.device("cpu"))
        for i in range(self.hparams.frames_out):
            print("Computing step: {}/{}\t".format(i + 1, self.frames_out), end="\r")
            y_hat = self(x)
            output_sequence[:, i] = y_hat.squeeze()
            x = torch.cat([x[:, -(self.hparams.frames_in - 1):], y_hat], dim=1)
        return output_sequence.squeeze()

    def log_step(self, y_hat_batch, y_batch, losses):
        # log losses
        run = ["val", "train"]
        for k, v in losses.items():
            self.logger.experiment.add_scalar("{}/".format(run[self.training]) + k, v, self.global_step)

        # log results
        names = ["v", "w", "u", "del_u", "j_ion"]
        y_hat_batch, y_batch = y_hat_batch.detach().cpu()[0], y_batch.detach().cpu()[0]
        for i in range(5):
            fig, ax = plot.compare(y_hat_batch[:, i], y_batch[:, i])
            self.logger.experiment.add_figure("{}/var_{}".format(run[self.training], names[i]), fig, self.global_step)
            plt.close(fig)

        # log parameter status
        if self.training:
            for k,v in self.named_parameters():
                self.logger.experiment.add_histogram(k, v, self.global_step)
        return

    def train_dataloader(self):
        train_fkset = ConcatSequence(self.root, self.hparams.frames_in, self.hparams.frames_out, self.hparams.step,
                                     transform=t.Compose([torch.as_tensor, Downsample(self.hparams.size)]),
                                     keys=self.hparams.train_search_regex, preload=True)
        logging.info("Found training dataset at: {}".format([x.filename for x in train_fkset.datasets]))
        train_loader = DataLoader(train_fkset, batch_size=self.hparams.batch_size, collate_fn=torch.stack,
                                  shuffle=True, drop_last=True, num_workers=self.n_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_fkset = ConcatSequence(self.root, self.hparams.frames_in, self.hparams.frames_out, self.hparams.step,
                                   transform=t.Compose([torch.as_tensor, Downsample(self.hparams.size)]),
                                   keys=self.hparams.val_search_regex, preload=True)
        logging.info("Found validation dataset at: {}".format([x.filename for x in val_fkset.datasets]))
        val_loader = DataLoader(val_fkset, batch_size=self.hparams.batch_size, collate_fn=torch.stack,
                                shuffle=False, drop_last=True, num_workers=self.n_workers, pin_memory=True)
        return val_loader


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # model args
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--n_filters', type=int, default=4)
    parser.add_argument('--n_hidden_filters', type=int, default=None)
    parser.add_argument('--kernel_size', type=int, nargs='+', default=(1, 7, 7))
    parser.add_argument('--n_latent_channels', type=int, default=4)
    parser.add_argument('--size', type=int, nargs='+', default=(256, 256))
    parser.add_argument('--frames_in', type=int, default=2)
    parser.add_argument('--frames_out', type=int, default=10)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--recon_loss', type=float, default=1.)
    parser.add_argument('--grad_loss', type=float, default=1.)
    parser.add_argument('--lr', type=float, default=0.001)

    # loader args
    parser.add_argument('--root', type=str, default="/media/ep119/DATADRIVE3/epignatelli/deepexcite/train_dev_set/")
    parser.add_argument('--train_search_regex', type=str, default="^spiral.*_PARAMS5.hdf5")
    parser.add_argument('--val_search_regex', type=str, default="^three_points.*_PARAMS5.hdf5")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=0)

    # trainer args
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--profile', default=False, action="store_true")
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--distributed_backend', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--logdir', type=str, default="logs/resnet/train")
    parser.add_argument('--min_step', type=int, default=None)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gradient_clip_val', type=float, default=0.)
    parser.add_argument('--track_gradient_norm', type=int, default=-1)

    args = parser.parse_args()

    # define loss weights
    loss_weights = {
        "recon_loss": args.recon_loss,
        "grad_loss": args.grad_loss,
    }

    # define model
    model = ResNet(n_layers=args.n_layers,
                   n_filters=args.n_filters,
                   n_hidden_filters=args.n_hidden_filters,
                   kernel_size=tuple(args.kernel_size),
                   n_latent_channels=args.n_latent_channels,
                   size=args.size,
                   frames_in=args.frames_in,
                   frames_out=args.frames_out,
                   step=args.step,
                   loss_weights=loss_weights,
                   lr=args.lr,
                   profile=args.profile,
                   root=args.root,
                   train_search_regex=args.train_search_regex,
                   val_search_regex=args.val_search_regex,
                   batch_size=args.batch_size,
                   n_workers=args.n_workers)

    # set logging level
    level = logging.DEBUG if args.debug or args.profile else logging.ERROR
    logging.basicConfig(stream=sys.stdout, level=level)

    # print debug info
    logging.info(model)

    # begin training
    trainer = Trainer.from_argparse_args(parser,
                                         fast_dev_run=args.debug,
                                         default_root_dir="logs/resnet/debug" if (args.profile or args.debug) else args.logdir,
                                         profiler=AdvancedProfiler("logs/resnet/debug/profile.txt") if args.profile else False,
                                         log_gpu_memory="all" if args.profile else None,
                                         train_percent_check=0.1 if args.profile else 1.0,
                                         val_percent_check=0.1 if args.profile else 1.0,
                                         checkpoint_callback=ModelCheckpoint(save_last=True, save_top_k=2),
                                         callbacks=[LearningRateLogger(), IncreaseFramsesOut(at_loss=3e-3 if not args.profile else 10.)])

    trainer.fit(model)
