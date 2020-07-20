import torch
from torch import nn


DEBUG = False

def log(*m):
    if DEBUG:
        print(*m)

def gradient(x):
    log("grad", x.shape)
    left = x
    right = nn.functional.pad(x, [0, 1, 0, 0])[..., :, 1:]
    top = x
    bottom = nn.functional.pad(x, [0, 0, 0, 1])[..., 1:, :]
    dx, dy = right - left, bottom - top
    dx[..., :, -1] = 0
    dy[..., -1, :] = 0
    log(dx.shape, dy.shape)
    return dx, dy   
        
    
def grad_mse_loss(gen_frames, gt_frames, reduction="sum"):
    grad_pred = gradient(gen_frames)
    grad_truth = gradient(gt_frames)

    # condense into one tensor and avg
    return nn.functional.mse_loss(grad_pred, grad_truth, reduction=reduction)


class Elu(nn.Module):
    def forward(self, x):
        return torch.nn.functional.elu(x)
    
    
class Downsample:
    def __init__(self, size, mode="bicubic"):
        self.size = size
        self.mode = mode
    
    def __call__(self, x):
        return nn.functional.interpolate(x, self.size)

    
class Normalise:
    def __call__(self, x):
        return (x - x.mean()) / x.std()
    