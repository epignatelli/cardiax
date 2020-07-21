import torch
from torch import nn
import random


DEBUG = False

def log(*m):
    if DEBUG:
        print(*m)

        
def time_grad(x):
    log("time grad", x.shape)
    past = x[:, :-1]
    future = x[:, 1:]
    return future - past


def time_grad_mse_loss(y_hat, y, reduction="sum"):
    time_grad_y_hat = time_grad(y_hat)
    time_grad_y = time_grad(y)
    return nn.functional.mse_loss(time_grad_y_hat, time_grad_y, reduction=reduction)

        
def space_grad(x):
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
        
    
def space_grad_mse_loss(gen_frames, gt_frames, reduction="sum"):
    grad_pred_x, grad_pred_y = space_grad(gen_frames)
    grad_truth_x, grad_truth_y = space_grad(gt_frames)
    grad_pred = torch.abs(grad_pred_x) + torch.abs(grad_pred_y)
    grad_truth = torch.abs(grad_truth_x) + torch.abs(grad_truth_y)

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
    
    
class Flip:
    def __call__(self, x):
        if random.random() > 0.5:
            x = torch.flip(x, dims=(random.randint(-2, -1), ))
        return x

    
class Rotate:
    def __call__(self, x):
        if random.random() > 0.5:
            x = torch.rot90(x, k=random.randint(1, 3), dims=(-2, -1))
        return x
    

class Flatten(nn.Module):	
    def forward(self, input):	
        return input.view(input.size(0), -1)	

    
class Unflatten(nn.Module):	
    def __init__(self, size=256, h=None, w=None):	
        self.size = size	
        self.h = h if h is not None else 1	
        self.w = w if w is not None else 1	
        super(Unflatten, self).__init__()	

    def forward(self, input):	
        return input.view(input.size(0), self.size, self.w, self.h)