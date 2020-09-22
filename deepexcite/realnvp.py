import logging
import sys
import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from sklearn import datasets


class RealNVP(LightningModule):
    def __init__(self, n_dims, n_hidden, n_steps, n_layers, batch_size):
        super().__init__()
        self.save_hyperparameters()

        # hyperparams
        self.ndim = n_dims
        self.n_hidden = n_hidden
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.batch_size = batch_size

        # compute checkboard mask
        mask = (torch.arange(n_dims) % 2).cuda()
        inverse_mask = torch.roll(mask, -1)
        logging.debug("mask = {}".format(mask))

        # base distribution
        # note that register_buffer is required to trasfer computation of methods from MultivariateGaussian to devices
        self.register_buffer("mu", torch.zeros(n_dims).cuda())
        self.register_buffer("logvar", torch.eye(n_dims).cuda())
        self.base_distribution = torch.distributions.MultivariateNormal(self.mu, self.logvar)

        self.transformations = nn.ModuleList()
        for i in range(n_layers):
            m = mask if i % 2 else inverse_mask
            self.transformations.append(Coupling(n_dims, n_hidden, n_steps, m))

    def forward(self, x):
        y = x
        for module in self.transformations:
            y = module(y)
        return y

    def inverse(self, y):
        sum_log_det_jac = torch.zeros(y.size(0)).type_as(y)
        x = y
        for module in reversed(self.transformations):  # reversed!
            x, log_det_jac = module.inverse(x)
            sum_log_det_jac.sub_(log_det_jac)
        return x, sum_log_det_jac

    def log_prob(self, x):
        z, logp = self.inverse(x)
        return self.base_distribution.log_prob(z) + logp

    def sample(self, n_samples):
        z = self.base_distribution.sample((n_samples,))
        x = self.forward(z)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def train_dataloader(self):
        dataset = datasets.make_moons(n_samples=2000, noise=.1)[0].astype("float32")
        print(dataset.shape)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=False)

    def training_step(self, batch, batch_idx):
        loss = -self.log_prob(batch).mean()
        return {"loss": loss}


class Coupling(nn.Module):
    def __init__(self, n_dims, n_hidden, n_steps, mask):
        super().__init__()
        # hparams
        self.n_dims = n_dims
        self.n_hidden = n_hidden
        self.n_steps = n_steps
        self.mask = mask

        # params
        self.scale = Affine(n_dims, n_hidden, n_steps)
        self.translate = Affine(n_dims, n_hidden, n_steps)

    def forward(self, x):
        y = x * self.mask
        s = self.scale(y) * (1 - self.mask)
        t = self.translate(y) * (1 - self.mask)
        y = y + (1 - self.mask) * (y * s.exp() + t)
        return y

    def inverse(self, y):
        x = y * self.mask
        s = self.scale(x) * (1 - self.mask)
        t = self.translate(x) * (1 - self.mask)
        x = (1 - self.mask) * (x - t) * (-s).exp() + x
        log_det_jac = torch.sum(s, dim=-1).exp()
        return x, log_det_jac


class Affine(nn.Sequential):
    def __init__(self, n_dims, n_hidden, n_steps):
        super().__init__()
        # entrance
        self.add_module("affine_0", nn.Linear(n_dims, n_hidden))
        self.add_module("affine_nonlin_0", Elu())

        # transform
        for i in range(n_steps):
            self.add_module("affine_{}".format(i + 1), nn.Linear(n_hidden, n_hidden))
            self.add_module("affine_nonlin_{}".format(i + 1), Elu())

        # exit
        self.add_module("affine_{}".format(n_steps + 1), nn.Linear(n_hidden, n_dims))
        self.add_module("affine_nonlin_{}".format(n_steps + 1), nn.Tanh())


class Elu(nn.Module):
    def forward(self, x):
        return nn.functional.elu(x)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # model args
    parser.add_argument('--n_dims', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=5)

    # loader args
    parser.add_argument('--batch_size', type=int, default=32)

    # trainer args
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--profile', default=False, action="store_true")
    parser.add_argument('--distributed_backend', type=str, default=None)
    parser.add_argument('--row_log_interval', type=int, default=10)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--logdir', type=str, default="logs/realnvp")
    parser.add_argument('--min_step', type=int, default=None)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=1000)

    # get arguments
    args = parser.parse_args()

    # set logging level
    level = logging.DEBUG if args.debug or args.profile else logging.ERROR
    logging.basicConfig(stream=sys.stdout, level=level)

    # build model
    model = RealNVP(args.n_dims,
                    args.n_hidden,
                    args.n_steps,
                    args.n_layers,
                    args.batch_size)
    logging.debug(model)

    # training
    trainer = Trainer.from_argparse_args(args,
                                         fast_dev_run=args.debug,
                                         default_root_dir="logs/debug" if (args.profile or args.debug) else args.logdir,
                                         profiler=args.profile,
                                         log_gpu_memory="all" if args.profile else None,
                                         train_percent_check=0.1 if args.profile else 1.0,
                                         val_percent_check=0.1 if args.profile else 1.0)

    trainer.fit(model)


if __name__ == "__main__":
    main()