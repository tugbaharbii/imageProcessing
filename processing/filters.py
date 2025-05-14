import cv2
import numpy as np

def mean_filter(image, kernel_size=3):
    """Apply a mean filter to the image (manual implementation)."""
    if len(image.shape) == 3:
        channels = cv2.split(image)
        filtered = [mean_filter_gray(c, kernel_size) for c in channels]
        return cv2.merge(filtered)
    else:
        return mean_filter_gray(image, kernel_size)

def mean_filter_gray(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='edge')
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.mean(region, dtype=np.float32)
    return out.astype(image.dtype)

def median_filter(image, kernel_size=3):
    """Apply a median filter to the image (manual implementation)."""
    if len(image.shape) == 3:
        channels = cv2.split(image)
        filtered = [median_filter_gray(c, kernel_size) for c in channels]
        return cv2.merge(filtered)
    else:
        return median_filter_gray(image, kernel_size)

def median_filter_gray(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='edge')
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.median(region)
    return out.astype(image.dtype) 

def edge_detection(image):
    """Kenar bulma filtresi (Sobel operatörü kullanarak) - manuel uygulama."""
    # Gri tonlamaya çevir
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Sobel operatörü için kerneller
    sobel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1], 
                         [0, 0, 0], 
                         [1, 2, 1]])
    
    # Görüntüyü kenarlar için hazırla
    padded = np.pad(gray, 1, mode='edge')
    grad_x = np.zeros_like(gray, dtype=np.float32)
    grad_y = np.zeros_like(gray, dtype=np.float32)
    
    # x ve y yönünde türev hesaplama
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            # Kernel bölgesini al
            region = padded[i:i+3, j:j+3]
            # Türevleri hesapla
            grad_x[i, j] = np.sum(region * sobel_x)
            grad_y[i, j] = np.sum(region * sobel_y)
    
    # Gradyan büyüklüğünü hesapla
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 0-255 aralığına normalize et
    normalized = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
    
    return normalized

def sharpening_filter(image):
    """Keskinleştirme filtresi - manuel uygulama."""
    # Gri tonlamaya çevir (gerekirse)
    if len(image.shape) == 3:
        channels = cv2.split(image)
        sharpened_channels = [sharpening_filter_gray(c) for c in channels]
        return cv2.merge(sharpened_channels)
    else:
        return sharpening_filter_gray(image)

def sharpening_filter_gray(image):
    # Keskinleştirme kerneli
    kernel = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
    
    # Görüntüyü kenarlar için hazırla
    padded = np.pad(image, 1, mode='edge')
    sharpened = np.zeros_like(image)
    
    # Keskinleştirme işlemi
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Kernel bölgesini al
            region = padded[i:i+3, j:j+3]
            # Keskinleştirme uygula
            value = np.sum(region * kernel)
            # Değeri sınırla
            sharpened[i, j] = np.clip(value, 0, 255)
    
    return sharpened.astype(image.dtype)

def smoothing_filter(image, kernel_size=5):
    """Yumuşatma filtresi (Gaussian benzeri) - manuel uygulama."""
    # Gri tonlamaya çevir (gerekirse)
    if len(image.shape) == 3:
        channels = cv2.split(image)
        smoothed_channels = [smoothing_filter_gray(c, kernel_size) for c in channels]
        return cv2.merge(smoothed_channels)
    else:
        return smoothing_filter_gray(image, kernel_size)

def smoothing_filter_gray(image, kernel_size=5):
    # Gaussian benzeri kernel oluştur
    mid = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Basit Gaussian ağırlıkları
    for i in range(kernel_size):
        for j in range(kernel_size):
            distance = np.sqrt((i - mid)**2 + (j - mid)**2)
            kernel[i, j] = np.exp(-(distance**2) / (2 * mid**2))
    
    # Kerneli normalize et
    kernel = kernel / np.sum(kernel)
    
    # Görüntüyü kenarlar için hazırla
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='edge')
    smoothed = np.zeros_like(image)
    
    # Filtreleme işlemi
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Kernel bölgesini al
            region = padded[i:i+kernel_size, j:j+kernel_size]
            # Ağırlıklı ortalama hesapla
            smoothed[i, j] = np.sum(region * kernel)
    
    return smoothed.astype(image.dtype) 