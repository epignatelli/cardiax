import logging
from glob import glob
import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
from dataset import ConcatSequence
import torchvision.transforms as t
from torch.utils.data import DataLoader
from torchvision.utils import make_grid as mg
from .utils import space_grad_mse_loss, energy_mse_loss
from .utils import Normalise, Rotate, Flip
import random
import math
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        logging.debug("x: {}".format(x.shape))
        dx = self.conv(x)
        dx = torch.nn.functional.elu(dx)
        logging.debug("conv: {}".format(dx.shape))
        return dx + x


class DeepReact(LightningModule):
    def __init__(self,
                 n_diffusion_layers,
                 n_diffusion_filters,
                 diffusion_kernel_size,
                 n_reaction_components,
                 n_reaction_layers,
                 n_reaction_filters,
                 reaction_kernel_size,
                 frames_in,
                 frames_out,
                 step,
                 root,
                 train_search_regex,
                 val_search_regex,
                 batch_size,
                 n_workers,
                 loss_weights=None,
                 lr=0.001,
                 profile=False):
        super().__init__()
        self.save_hyperparameters()

        # public:
        self.n_diffusion_layers = n_diffusion_layers
        self.n_diffusion_filters = n_diffusion_filters
        self.diffusion_kernel_size = diffusion_kernel_size
        self.n_reaction_components = n_reaction_components
        self.n_reaction_layers = n_reaction_layers
        self.n_reaction_filters = n_reaction_filters
        self.reaction_kernel_size = reaction_kernel_size
        self.frames_in = frames_in
        self.step = step
        self.loss_weights = loss_weights or {}
        self.lr = lr
        self.profile = profile
        self.root = root
        self.train_search_regex = train_search_regex
        self.val_search_regex = val_search_regex
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.frames_out = frames_out

        # private:
        self._log_gpu_mem_step = 0

        # diffusion net
        diffusion_padding = tuple([math.floor(x / 2) for x in diffusion_kernel_size])
        self.diffusion_inlet = nn.Conv3d(self.frames_in, n_diffusion_filters, kernel_size=diffusion_kernel_size, stride=1, padding=diffusion_padding)
        self.diffusion_flow = nn.ModuleList([
            ResidualBlock(n_diffusion_filters, n_diffusion_filters, kernel_size=diffusion_kernel_size, stride=1, padding=diffusion_padding)
            for i in range(n_diffusion_filters)
        ])
        self.diffusion_outlet = nn.Conv3d(n_diffusion_filters, 1, kernel_size=diffusion_kernel_size, stride=1, padding=diffusion_padding)

        # reaction net
        reaction_padding = tuple([math.floor(x / 2) for x in reaction_kernel_size])
        self.reaction_inlet = nn.Conv3d(self.frames_in + 1, n_reaction_filters, kernel_size=reaction_kernel_size, stride=1, padding=reaction_padding)
        self.reaction_flow = nn.ModuleList([
            ResidualBlock(n_reaction_filters, n_reaction_filters, kernel_size=reaction_kernel_size, stride=1, padding=reaction_padding)
            for i in range(n_reaction_layers)
        ])
        self.diffusion_outlet = nn.Conv3d(n_reaction_filters, self.n_reaction_components, kernel_size=reaction_kernel_size, stride=1, padding=reaction_padding)

    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())

    def diffusion(self, x):
        d_hat = self.diffusion_inlet(x)
        for m in self.diffusion_flow:
            d_hat = m(d_hat)
        d_hat = self.diffusion_outlet(d_hat)
        return d_hat

    def reaction(self, x, diffusion):
        x = torch.cat([x, diffusion])
        r_hat = self.reaction_inlet(x)
        for m in self.reaction_flow:
            r_hat = m(r_hat)
        r_hat = self.reaction_outlet(r_hat)
        return r_hat

    def forward(self, x):
        d_hat = self.diffusion(x)
        r_hat = self.reaction(x, d_hat)
        return d_hat, r_hat

    def backward(self, *args):
        # we're using truncated backprop through time where the inputs depend on new outputs
        # the backward methods is called inside training_step
        # we override this to do nothing
        return

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
        memory = torch.cuda.memory_allocated(self.diffusion_inlet.bias.device) / 1024 / 1024
        self.logger.experiment.add_scalar("gpu_mem", memory, self._log_gpu_mem_step)
        logging.debug("gpu_memory {}".format(memory))
        self._log_gpu_mem_step += 1
        return

    def on_train_start(self):
        # add graph with a randomly-sized input
        self.logger.experiment.add_graph(self, torch.randn((16, self.frames_in, 3, 256, 256), device=self.diffusion_inlet.bias.device))
        return

    def training_step(self, batch, batch_idx):
        x, y = batch.split((self.frames_in, self.frames_out), dim=1)
        self.profile_gpu_memory()

        output_sequence = torch.empty_like(y, requires_grad=False)
        loss = {}
        for i in range(self.frames_out):
            # forward pass
            d_hat, r_hat = self(x)

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
        for i, module in enumerate(self.diffusion_flow):
            # conv kernels
            self.logger.experiment.add_image("kernel/layer-{}".format(i), mg(module.conv.weight[0], nrow=self.n_filters, normalize=True), self.current_epoch)
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
        training_keys = [os.path.basename(f) for f in glob(self.root + os.sep + self.train_search_regex)]
        train_fkset = ConcatSequence(self.root, self.frames_in, self.frames_out, self.step, transform=train_transform, squeeze=True, keys=training_keys, clean_from_stimuli=self.clean_from_stimuli)
        train_loader = DataLoader(train_fkset, batch_size=self.batch_size, collate_fn=torch.stack, shuffle=True, drop_last=True, num_workers=self.n_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_transform = t.Compose([torch.as_tensor, Normalise()])
        val_keys = [os.path.basename(f) for f in glob(self.root + os.sep + self.val_search_regex)]
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
    parser.add_argument('--n_diffusion_layers', type=int, default=10)
    parser.add_argument('--n_diffusion_filters', type=int, default=4)
    parser.add_argument('--diffusion_kernel_size', type=int, nargs='+', default=(1, 7, 7))
    parser.add_argument('--n_reaction_layers', type=int, default=10)
    parser.add_argument('--n_reaction_filters', type=int, default=4)
    parser.add_argument('--reaction_kernel_size', type=int, nargs='+', default=(1, 7, 7))
    parser.add_argument('--n_reaction_components', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--recon_loss', type=float, default=1.)
    parser.add_argument('--space_grad_loss', type=float, default=1.)
    parser.add_argument('--time_grad_loss', type=float, default=1.)
    parser.add_argument('--energy_loss', type=float, default=0.)

    # loader args
    parser.add_argument('--root', type=str, default="/media/ep119/DATADRIVE3/epignatelli/deepexcite/train_dev_set/")
    parser.add_argument('--paramset', type=str, default="3")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=0)

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

    # define loss weights
    loss_weights = {
        "recon_loss": args.recon_loss,
        "space_grad_loss": args.space_grad_loss,
        "energy_loss": args.energy_loss,
        "time_grad_loss": args.time_grad_loss
    }

    # define model
    model = DeepReact(n_diffusion_layers=args.n_diffusion_layers,
                      n_diffusion_filters=args.n_diffusion_filters,
                      diffusion_kernel_size=tuple(args.diffusion_kernel_size),
                      n_reaction_layers=args.n_reaction_layers,
                      n_reaction_filters=args.n_reaction_filters,
                      reaction_kernel_size=tuple(args.diffusion_kernel_size),
                      n_reaction_components=args.n_reaction_components,
                      frames_in=args.frames_in, frames_out=args.frames_out, step=args.step,
                      loss_weights=loss_weights,
                      lr=args.lr,
                      profile=args.profile,
                      root=args.root,
                      paramset=args.paramset,
                      batch_size=args.batch_size,
                      n_workers=args.n_workers)

    # set logging level
    level = logging.DEBUG if args.debug or args.profile else logging.ERROR
    logging.basicConfig(stream=sys.stdout, level=level)

    # print debug info
    logging.debug(model)
    logging.debug("parameters: {}".format(model.parameters_count()))

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
