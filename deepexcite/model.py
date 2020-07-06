import torch
from torch import nn
import torchvision
import fk
import h5py
import matplotlib.pyplot as plt

class DeepExcite(nn.Module):
    def __init__(self)
class ConvVAE(nn.Module):
    def __init__(self, depth, latent_dim=8):
        super(ConvVAE, self).__init__()
        filters = [3, 8, 16, 32]
        self.encoder = Encoder(filters, 3, 3, 0)
        
        self.logvar = nn.Linear(512, latent_dim)
        self.mu = nn.Linear(512, latent_dim)
        self.latent = nn.Linear(1, latent_dim)
    
        self.decoder = Decoder(reversed(filters), 3, 3, 0)
        return
        
    def reparameterise(self, logvar, mu):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.nn.Parameter(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, X):
        encoded = self.encoder(X)
        print(encoded.shape)
        encoded = encoded.view(1, -1)
        print(encoded.shape)
        logvar = self.logvar(encoded)
        mu = self.mu(encoded)
        z = self.reparameterise(logvar, mu)
        print(z.shape)
        z = z.view(1, 32, 4, 4)
        decoded = self.decoder(z)
        return decoded, logvar, mu


class Encoder(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        assert len(channels) == 4
        self.encoder = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size, stride, padding),
            Elu(),  # strictly monotonic
            nn.Conv2d(channels[1], channels[2], kernel_size, stride, padding),
            Elu(),
            nn.Conv2d(channels[2], channels[3], kernel_size, stride, padding),
            Elu(),
        )

        
class Decoder(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, output_size=None):
        assert len(channels) == 4
        self.output_size = output_size
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(channels[0], channels[1], 3, 3, 0),
            nn.Conv2d(channels[1], channels[1], 3, 3, 0),
            Elu(),
            nn.Conv2d(channels[1], channels[2], 3, 3, 0),
            nn.ConvTranspose2d(channels[2], channels[2], 3, 3, 0),
            Elu(),
            nn.Conv2d(channels[2], channels[3], 3, 3, 0),
            nn.ConvTranspose2d(channels[3], channels[3], 3, 3, 0),
        ])
        return
    
    def forward(self, X):
        for i in range(len(decoder) - 1):
            X = decoder[i](X)
        y_hat = decoder[i + 1](X, output_size=self.output_size)
        return torch.sigmoid(y_hat)

# class DeepExcite(nn.Module):
#     def __init__(self, channels_in=5, channels_out=10, loss=None):
#         super(DeepExcite, self).__init__()
#         self.encoder = Encoder(channels_in)
#         self.inference = Inference(n_states_in)
#         self.generative = Generative(32, n_states_out)
#         self.get_loss = loss or torch.nn.functional.mse_loss
        
#     def forward(self, X, Y=None):
#         z = self.encoder(X)
#         h, z = self.inference(z)
#         new_states = self.generative(z)
        
#         loss = None
#         if Y is not None and self.training:
#             loss = self.get_loss(new_states, Y)
#         return new_states, loss
        
            
# class Generative(nn.Module):
#     def __init__(self, ndim, channels_out):
#         super(Generative, self).__init__()
#         pass


# class Inference(nn.Module):
#     def __init__(self, flow_steps, flow="iaf"):
#         super(Inference, self).__init__()
#         self.flow = nn.ModuleList()
#         for i in range(flow_steps):
#             if flow == "iaf":
#                 self.flow.append(InverseAutoregressiveFlow(ndim))
#             elif flow == "ddsf":
#                 self.flow.append(DeepDenseSigmoidalFlow(ndim))
#             else:
#                 raise TypeError("Unknown flow of type {}".format(flow))

#     def reparameterise(self, sigma, mu):
#         return
                
#     def forward(self, h, z):
#         return      


# class Encoder(nn.Module):
#     def __init__(self, channels_in):
#         super(Encoder, self).__init__()
#         self.encoder = nn.ModuleList([
#             ConvDown3d(channels_in, 8, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
#             ConvDown3d(8, 16, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
#             ConvDown3d(16, 32, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
#             ])

#     def forward(self, X):
#         for module in self.encoder:
#             X = module(X)
#         return h, z
        
        
# class InverseAutoregressiveFlow(nn.Module):
#     def __init__(self, ndim):
#         super(InverseAutoregressiveFlow, self).__init__()
#         return

    
# class DeepDenseSigmoidalFlow(nn.Module):
#     def __init__(self, ndim):
#         super(DeepDenseSigmoidalFlow, self).__init__()
#         return

    
# class ConvDown3d(nn.Module):
#     def __init__(self, channels_in, channels_out, kernel_size=(2, 3, 3), stride=(1, 2, 2),
#                  pool_kernel=(2, 3, 3), pool_stride=(1, 2, 2)):
#         super(ConvDown3d, self).__init__()
#         self.down = nn.ModuleList([
#             nn.Conv3d(channels_in, channels_out, kernel_size=kernel_size, stride=stride),
#             Relu(),
#             nn.Conv3d(channels_out, channels_out, kernel_size=kernel_size, stride=stride),
#             Relu(),
#             nn.MaxPool3d(pool_kernel, pool_stride)
#         ])
        
#     def forward(self, X):
#         for layer in down:
#             X = layer(X)
#         return X 


# class ConvUp3d(nn.Module):
#     def __init__(self, channels_in, channels_out, kernel_size=(2, 3, 3), stride=(1, 2, 2),
#                  up_kernel=(2, 3, 3), up_stride=(1, 2, 2)):
#         super(ConvUp3d, self).__init__()
#         self.down = nn.ModuleList([
#             nn.Conv3d(channels_in, channels_out, kernel_size=kernel_size, stride=stride),
#             Relu(),
#             nn.Conv3d(channels_out, channels_out, kernel_size=kernel_size, stride=stride),
#             Relu(),
#             nn.ConvTranspose3d(channels_in, channels_out, kernel_size=up_kernel, stride=up_stride)
#         ])
        
#     def forward(self, X):
#         for layer in down:
#             X = layer(X)
#         return X

class Relu(nn.Module):
    def forward(self, X):
        return nn.functional.relu(X)
    

class Elu(nn.Module):
    def forward(self, X):
        return nn.functional.elu(X)