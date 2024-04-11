import torch
import numpy as np
from sklearn.linear_model import LinearRegression

def autocorrelation(arrs, max_lag=25):
    """Compute the spatial autocorrelation of a list of 3D arrays.

    Args:
        arrs: list of 3D arrays or a single 3D array
        max_lag: int, the maximum lag to compute the autocorrelation for in all three dimensions

    Returns:
        result: 3D tensor, the autocorrelation of the arrays across depth, height, and width
    """
    if not isinstance(arrs, list):
        arrs = [arrs]

    covar = torch.zeros((max_lag, max_lag, max_lag))
    covar_denom = torch.zeros((max_lag, max_lag, max_lag))
    var = 0
    var_denom = 0

    for a in arrs:
        a = a - a.mean()
        for d in range(max_lag):  # Depth iteration
            for i in range(max_lag):  # Height iteration
                for j in range(max_lag):  # Width iteration
                    valid_area = a[..., :a.shape[-3] - d, :a.shape[-2] - i, :a.shape[-1] - j]
                    shifted_area = a[..., d:, i:, j:]
                    c = (valid_area * shifted_area).sum()
                    n = valid_area.numel()
                    covar[d, i, j] += c
                    covar_denom[d, i, j] += n
        var += (a ** 2).sum()
        var_denom += a.numel()

    covar = covar / covar_denom
    var = var / var_denom

    ac = covar / var
    return ac


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, images, n_iters=1, transform=None):
        self.images = images
        self.n_images = len(images)
        self.n_iters = n_iters
        self.transform = transform

    def __len__(self):
        return self.n_images * self.n_iters

    def __getitem__(self, idx):
        idx = idx % self.n_images
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
    

class PredictDataset(torch.utils.data.Dataset):

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image


def minimise_mse(x, y):
    x_ = x.flatten().reshape(-1, 1)
    y_ = y.flatten().reshape(-1, 1)

    reg = LinearRegression().fit(x_, y_)
    a = reg.coef_
    b = reg.intercept_
    return a * x + b


def normalise(x):
    low = np.percentile(x, 0.1)
    high = np.percentile(x, 99.9)
    x = (x - low) / (high - low)
    return x
