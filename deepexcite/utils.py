import torch
from torch import nn


# @torch.jit.script
def energy_mse_loss(y_hat, y):
    y_hat_energy = y_hat.sum(dim=(-3, -2, -1))
    y_energy = y.sum(dim=(-3, -2, -1))
    return torch.sqrt(nn.functional.mse_loss(y_hat_energy, y_energy, reduction="mean"))


# @torch.jit.script
def time_grad(x):
    past = x[:, :-1]
    future = x[:, 1:]
    return future - past


# @torch.jit.script
def time_grad_mse_loss(y_hat, y):
    time_grad_y_hat = time_grad(y_hat)
    time_grad_y = time_grad(y)
    return torch.sqrt(nn.functional.mse_loss(time_grad_y_hat, time_grad_y, reduction="mean"))


# @torch.jit.script
def space_grad(x):
    left = x
    right = nn.functional.pad(x, [0, 1, 0, 0])[..., :, 1:]
    top = x
    bottom = nn.functional.pad(x, [0, 0, 0, 1])[..., 1:, :]
    dx, dy = right - left, bottom - top
    dx[..., :, -1] = 0
    dy[..., -1, :] = 0
    return dx, dy


# @torch.jit.script
def space_grad_mse_loss(gen_frames, gt_frames):
    grad_pred_x, grad_pred_y = space_grad(gen_frames)
    grad_truth_x, grad_truth_y = space_grad(gt_frames)
    grad_pred = torch.abs(grad_pred_x) + torch.abs(grad_pred_y)
    grad_truth = torch.abs(grad_truth_x) + torch.abs(grad_truth_y)

    # condense into one tensor and avg
    return nn.functional.mse_loss(grad_pred, grad_truth, reduction="mean")


# @torch.jit.script
def Normalise(x):
    return (x - x.mean()) / x.std()


# @torch.jit.script
def Flip(x):
    if torch.rand(1) > 0.5:
        x = torch.flip(x, dims=([int(torch.randint(low=-2, high=-1, size=(1, )).item())]))
    return x


# @torch.jit.script
def Rotate(x):
    if torch.rand(1) > 0.5:
        x = torch.rot90(x, k=(int(torch.randint(low=-2, high=-1, size=(1, )).item())))
    return x


class Noise:
    def __init__(self, frames_in, scale=0.1):
        self.frames_in = frames_in
        self.scale = scale

    def __call__(self, x):
        if random.random() > 0.5:
            x[:self.frames_in] = x[:self.frames_in] + torch.empty(x[:self.frames_in].shape).normal_(mean=x.mean(), std=1) * self.scale
        return x


class Downsample:
    def __init__(self, size, mode="bicubic"):
        self.size = size
        self.mode = mode

    def __call__(self, x):
        return nn.functional.interpolate(x, self.size)
