import torch
from torch import nn
import torchvision
import fk
import h5py
import matplotlib.pyplot as plt


class DeepExcite(nn.Module):
    def __init__(self, n_states_in=5, n_states_out=10):
        super(DeepExcite, self).__init__()
        self.states_encoder = Encoder(n_states_in)
        self.stimuli_encoder = Encoder(n_states_in)
        self.states_decoder = Decoder(64, n_states_out)
        
    def forward(self, X_state, X_stim, Y_state=None):
        states = self.states_encoder(X_state)
        stimuli = self.stimuli_encoder(X_stim)
        latent = torch.cat([states, stimuli], dim=1)
        new_states = self.states_decoder(latent)
        if Y_state is not None:
            loss = torch.nn.functional.mse_loss(new_states, Y_state)
            return new_states, loss
        return new_states

class StochasticInference(nn.Module):
    def __init__(self, n_states_in, n_states_out):
        super(StochasticInference, self).__init__()
        self.u_encoder = Encoder(n_states_in, n_states_out)
        self.v_encoder = Encoder(n_states_in, n_states_out)
        self.w_encoder = Encoder(n_states_in, n_states_out)
        self.mu = nn.Linear(32, 1)
        self.logvar = nn.Linear(32, 1)

        # init with normal weights
        torch.nn.init.normal(self.mu)
        torch.nn.init.normal(self.logvar)
        return

    @staticmethod
    def kl_divergence(z, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    @staticmethod
    def reparameterize(mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def get_loss(self, z, mu, logvar):
        loss = self.kl_divergence(z, mu, logvar)
        return loss

    def forward(self, X):
        X = self.encode(X)
        mu = self.mu(X)
        logvar = self.logvar(X)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Encoder(nn.Module):
    def __init__(self, channels_in):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList([
            nn.Conv2d(channels_in, 8, kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            ])

    def forward(self, X):
        for module in self.encoder:
            X = module(X)
        return X


class Decoder(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(channels_in, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2)),
            nn.ConvTranspose2d(8, channels_out, kernel_size=(3, 3), stride=(2, 2))
            ])

    def forward(self, X):
        for module in self.encoder:
            X = module(X)
        return X
