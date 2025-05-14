import numpy as np

def rotate_image(image, angle):
    if angle == 90:
        return np.rot90(image, k=1)
    elif angle == 180:
        return np.rot90(image, k=2)
    elif angle == 270:
        return np.rot90(image, k=3)
    else:
        raise ValueError("Sadece 90, 180, 270 derece destekleniyor.")

def flip_image(image, mode='horizontal'):
    if mode == 'horizontal':
        return np.fliplr(image)
    elif mode == 'vertical':
        return np.flipud(image)
    else:
        raise ValueError("mode 'horizontal' veya 'vertical' olmalı.")