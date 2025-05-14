import numpy as np

def dilation(image, kernel_size=3):
    # Binary görüntü için manuel dilation
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.max(region)
    return out

def erosion(image, kernel_size=3):
    # Binary görüntü için manuel erosion
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=255)
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.min(region)
    return out