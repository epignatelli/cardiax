import logging
import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from dataset import ConcatSequence
import torchvision.transforms as t
from torch.utils.data import DataLoader
from torchvision.utils import make_grid as mg
from .utils import space_grad_mse_loss, energy_mse_loss
from .utils import Normalise, Rotate, Flip
import random
import math
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
import torch.distributed as dist


class IncreaseFramsesOut(Callback):
    def __init__(self, monitor="loss", at_loss=None, every_k_epochs=None, max_value=20):
        self.monitor = monitor
        self.at_loss = at_loss
        self.every_k_epochs = every_k_epochs
        self.max_value = max_value

    def on_epoch_end(self, trainer, pl_module):
        # get final loss (if ddp, the model we're in)
        loss = trainer.callback_metrics.get(self.monitor)
        if loss is None:
            print("\nWARNING: IncreaseFramesOut callback failed. Cannot retrieve metric {}".format(self.monitor))
            return

        # synch and average if ddp
        dist.all_reduce(loss)
        loss /= abs(dist.get_world_size())
        print("\nIncreaseFramsesOut callback: loss is {}".format(loss))

        if (self.at_loss is not None and loss <= self.at_loss) or (self.every_k_epochs is not None and trainer.current_epoch % self.every_k_epochs):
            if pl_module.frames_out >= self.max_value:
                print("\nEpoch\t{} - IncreaseFramsesOut callback: Max number of output frames reached")
                return
            pl_module.frames_out += 1
            trainer.train_dataloader.dataset.frames_out = pl_module.frames_out
            trainer.val_dataloaders[0].dataset.frames_out = pl_module.frames_out
            assert pl_module.frames_out == trainer.train_dataloader.dataset.frames_out == trainer.val_dataloaders[0].dataset.frames_out
            print("\nEpoch\t{} - IncreaseFramsesOut callback: Increase number of output frames")
        else:
            print("\nEpoch\t{} - IncreaseFramsesOut callback: Do not increase the number of output frames")

        print("\nIncreaseFramsesOut callback: model.frames_out: {}; train_loader.frames_out: {}; val_loader.frames_out: {}".format(
              pl_module.frames_out, trainer.train_dataloader.dataset.frames_out, trainer.val_dataloaders[0].dataset.frames_out))
        # log event
        pl_module.logger.experiment.add_scalar("frames_out", pl_module.frames_out, trainer.current_epoch + 1)
        return


class SplineActivation(nn.Module):
    def __init__(self, degree):
        super(SplineActivation, self).__init__()
        self.weights = nn.Parameter(torch.empty(degree + 1))

    def forward(self, x):
        for i in range(len(self.weights)):
            x = x + (x ** i) * self.weights[i]
        return x


class StochasticConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.logvar = nn.Linear(torch.prod(kernel_size) * in_channels, torch.prod(kernel_size) * out_channels)
        self.mu = nn.Linear(torch.prod(kernel_size) * in_channels, torch.prod(kernel_size) * out_channels)

    def reparameterise(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps

    def get_loss(self, mu, logvar):
        # elbo
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, z, x):
        logvar = self.logvar(z)
        mu = self.mu(z)
        kernel = self.reparameterise(mu, logvar).view(self.kernel_size)
        # x dim is (batch, sequence_length, channels, width, height)
        y_hat = nn.functional.conv3d(x, kernel, bias=None, stride=self.stride, padding=self.padding)
        return y_hat


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
        logging.debug("x: {}".format( x.shape))
        dx = self.conv(x)
        dx = self.activation(dx)
        log("conv: ", dx.shape)
        if self.attention is not None:
            a = self.attention(x)
            log("attention: ", a.shape)
            return (dx * a) + x
        return dx + x


class ResNet(LightningModule):
    def __init__(self,
                 n_layers,
                 n_filters,
                 kernel_size,
                 residual_step,
                 activation,
                 frames_in,
                 frames_out,
                 step,
                 root,
                 paramset,
                 batch_size,
                 n_workers,
                 loss_weights={},
                 attention="self",
                 lr=0.001,
                 profile=False,
                 clean_from_stimuli=False):
        super().__init__()
        self.save_hyperparameters()

        # public:
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.residual_step = residual_step
        self.activation = activation
        self.frames_in = frames_in
        self.step = step
        self.loss_weights = loss_weights
        self.attention = attention
        self.lr = lr
        self.profile = profile
        self.root = root
        self.paramset = paramset
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.clean_from_stimuli = clean_from_stimuli
        self.frames_out = frames_out

        # private:
        self._val_steps_done = 0
        self._log_gpu_mem_step = 0

        # init parameters
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
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimisers[0], verbose=True, min_lr=1e-4),
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
        x, y = batch.split((self.frames_in, self.frames_out), dim=1)
        self.profile_gpu_memory()

        output_sequence = torch.empty_like(y, requires_grad=False)
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
        return {"loss": total_loss, "log": logs}#, "x": batch[:, :self.frames_in], "y_hat": output_sequence, "y": y}

    def training_epoch_end(self, outputs):
        i = random.randint(0, len(outputs) - 1)
        return {"loss": torch.stack([x["loss"] for x in outputs]).mean()}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch.split((self.frames_in, self.frames_out), dim=1)
        self.profile_gpu_memory()

        output_sequence = torch.empty_like(y, requires_grad=False)
        loss = {}
        for i in range(self.frames_out):
            # forward pass
            y_hat = self(x).squeeze()
            self.profile_gpu_memory()

            # calculate loss
            current_loss = self.get_loss(y_hat, y[:, i])
            total_loss = sum(current_loss.values())
            for k, v in current_loss.items():
                loss.update({k: (loss.get(k, 0.) + v)})
            self.profile_gpu_memory()

            # update output sequence
            output_sequence[:, i] = y_hat
            self.profile_gpu_memory()

            # update input sequence with predicted frames
            if (self.frames_out > 1):
                x = torch.cat([x[:, -(self.frames_in - 1):], y_hat.unsqueeze(1)], dim=1)
                self.profile_gpu_memory()

        # logging losses
        logs = {"val_loss/" + k: v for k, v in loss.items()}
        logs["val_loss/total_loss"] = total_loss
        return {"loss": total_loss, "log": logs}#, "x": batch[:, :self.frames_in], "y_hat": output_sequence, "y": y}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"val_loss/total_loss": loss}
        return {"val_loss": loss, "log": logs}

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

    @torch.no_grad()
    def infer(self, x):
        if x.dim() == 4:  # no batch dimension
            x = x.unsqueeze(0)
        output_sequence = torch.empty((1, self.frames_out, 3, x.size(-1), x.size(-1)), device=torch.device("cpu"))
        for i in range(self.frames_out):
            print("Computing step: {}/{}\t".format(i + 1, self.frames_out), end="\r")
            y_hat = self(x)
            output_sequence[:, i] = y_hat.squeeze()
            x = torch.cat([x[:, -(self.frames_in - 1):], y_hat], dim=1)
        return output_sequence.squeeze()

    def log_images(self, x, y_hat, y, normalise=True, nrow=10):
        i = random.randint(0, y_hat.size(0) - 1)
        self.logger.experiment.add_image("val_w/input", mg(x[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_v/input", mg(x[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_u/input", mg(x[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_w/pred", mg(y_hat[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_w/truth", mg(y[i, :, 0].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_v/pred", mg(y_hat[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_v/truth", mg(y[i, :, 1].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_u/pred", mg(y_hat[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        self.logger.experiment.add_image("val_u/truth", mg(y[i, :, 2].unsqueeze(1), nrow=nrow, normalize=normalise), self.current_epoch)
        return

    def train_dataloader(self):
        train_transform = t.Compose([torch.as_tensor, Normalise(), Rotate(), Flip()])
        training_keys = ["spiral_params{}.hdf5".format(self.paramset), "three_points_params{}.hdf5".format(self.paramset)]
        train_fkset = ConcatSequence(self.root, self.frames_in, self.frames_out, self.step, transform=train_transform, squeeze=True, keys=training_keys, clean_from_stimuli=self.clean_from_stimuli)
        train_loader = DataLoader(train_fkset, batch_size=self.batch_size, collate_fn=torch.stack, shuffle=True, drop_last=True, num_workers=self.n_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_transform = t.Compose([torch.as_tensor, Normalise()])
        val_keys = ["heartbeat_params{}.hdf5".format(self.paramset)]
        val_fkset = ConcatSequence(self.root, self.frames_in, self.frames_out, self.step, transform=val_transform, squeeze=True, keys=val_keys, clean_from_stimuli=self.clean_from_stimuli)
        val_loader = DataLoader(val_fkset, batch_size=self.batch_size, collate_fn=torch.stack, drop_last=True, num_workers=self.n_workers, pin_memory=True)
        return val_loader


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
    parser.add_argument('--paramset', type=str, default="3")
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--clean_from_stimuli', default=False, action="store_true")

    # trainer args
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--profile', default=False, action="store_true")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--distributed_backend', type=str, default=None)
    parser.add_argument('--row_log_interval', type=int, default=10)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--logdir', type=str, default="logs")
    parser.add_argument('--min_step', type=int, default=None)
    parser.add_argument('--precision', type=int, default=32)

    args = parser.parse_args()
    utils.DEBUG = DEBUG = args.debug  # hacky, TODO(epignatelli)

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
                   profile=args.profile,
                   root=args.root,
                   paramset=args.paramset,
                   batch_size=args.batch_size,
                   n_workers=args.n_workers,
                   clean_from_stimuli=args.clean_from_stimuli)

    # print debug info
    log(model)
    log("parameters: {}".format(model.parameters_count()))

    # begin training
    trainer = Trainer.from_argparse_args(parser,
                                         fast_dev_run=args.debug,
                                         default_root_dir="logs/debug" if (args.profile or args.debug) else args.logdir,
                                         profiler=args.profile,
                                         log_gpu_memory="all" if args.profile else None,
                                         train_percent_check=0.1 if args.profile else 1.0,
                                         val_percent_check=0.1 if args.profile else 1.0,
                                         checkpoint_callback=ModelCheckpoint(save_last=True, save_top_k=2),
                                         callbacks=[LearningRateLogger(), IncreaseFramsesOut(every_k_epochs=5 if not args.profile else 10.)])

    trainer.fit(model)
