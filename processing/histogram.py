import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_histogram(image, title="Histogram"):
    plt.figure(figsize=(10, 5))
    
    if len(image.shape) == 3:  # RGB image
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title("RGB Histogram")
    else:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='gray')
        plt.title("Grayscale Histogram")
        
    plt.xlim([0, 256])
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def histogram_equalization(image):
    if len(image.shape) == 3:
        # RGB image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        # Grayscale image
        return cv2.equalizeHist(image)

def contrast_stretching(image, min_out=0, max_out=255):
    """
    Kontrast germe - resimdeki piksel değerlerini belirli bir aralığa yayar.
    Elle yazılmış uygulama, hazır kütüphane fonksiyonu kullanılmamıştır.
    
    Parametreler:
    - image: Giriş görüntüsü
    - min_out: Çıkış piksel değeri alt sınırı (varsayılan: 0)
    - max_out: Çıkış piksel değeri üst sınırı (varsayılan: 255)
    
    Dönüş:
    - stretched_image: Kontrast genişletilmiş görüntü
    """
    # Çok kanallı görüntü için
    if len(image.shape) == 3:
        # BGR kanallarını ayır
        b, g, r = cv2.split(image)
        
        # Her kanalı ayrı ayrı işle
        b_stretched = contrast_stretch_channel(b, min_out, max_out)
        g_stretched = contrast_stretch_channel(g, min_out, max_out)
        r_stretched = contrast_stretch_channel(r, min_out, max_out)
        
        # Kanalları tekrar birleştir
        return cv2.merge([b_stretched, g_stretched, r_stretched])
    else:
        # Tek kanallı (gri tonlamalı) görüntü için
        return contrast_stretch_channel(image, min_out, max_out)

def contrast_stretch_channel(channel, min_out=0, max_out=255):
    """Tek bir kanal için kontrast germe işlemi."""
    # Görüntünün min ve max değerlerini bul
    min_val = np.min(channel)
    max_val = np.max(channel)
    
    # Min ve max değerler aynıysa (tek renk görüntü), bir değişiklik yapma
    if min_val == max_val:
        return channel.copy()
    
    # Yeni bir görüntü oluştur
    stretched = np.zeros_like(channel, dtype=np.float32)
    
    # Kontrast germe formülü: yeni_piksel = (piksel - min) * (max_out - min_out) / (max - min) + min_out
    stretched = (channel.astype(np.float32) - min_val) * (max_out - min_out) / (max_val - min_val) + min_out
    
    # Değerleri sınırla ve uint8'e dönüştür
    return np.clip(stretched, min_out, max_out).astype(np.uint8)

def contrast_spreading(image, percentage=5):
    """
    Kontrast yayma - histogramın en düşük ve en yüksek değerlerini kesip, 
    kalan değerleri tüm aralığa yayar.
    Elle yazılmış uygulama, hazır kütüphane fonksiyonu kullanılmamıştır.
    
    Parametreler:
    - image: Giriş görüntüsü
    - percentage: Histogramın başından ve sonundan kesilecek yüzde (varsayılan: 5)
    
    Dönüş:
    - spread_image: Kontrast yayılmış görüntü
    """
    # Çok kanallı görüntü için
    if len(image.shape) == 3:
        # BGR kanallarını ayır
        b, g, r = cv2.split(image)
        
        # Her kanalı ayrı ayrı işle
        b_spread = contrast_spread_channel(b, percentage)
        g_spread = contrast_spread_channel(g, percentage)
        r_spread = contrast_spread_channel(r, percentage)
        
        # Kanalları tekrar birleştir
        return cv2.merge([b_spread, g_spread, r_spread])
    else:
        # Tek kanallı (gri tonlamalı) görüntü için
        return contrast_spread_channel(image, percentage)

def contrast_spread_channel(channel, percentage=5):
    """Tek bir kanal için kontrast yayma işlemi."""
    # Histogramı hesapla (elle)
    hist = np.zeros(256, dtype=np.int32)
    for val in channel.flatten():
        hist[val] += 1
    
    # Kümülatif dağılımı hesapla
    cumsum = np.cumsum(hist)
    
    # Toplam piksel sayısı
    total_pixels = channel.size
    
    # Alt ve üst eşik değerlerini belirle
    min_thresh = total_pixels * (percentage / 100.0)
    max_thresh = total_pixels * (1 - percentage / 100.0)
    
    # Eşik değerlerine karşılık gelen piksel değerlerini bul
    min_val = 0
    max_val = 255
    
    for i in range(256):
        if cumsum[i] >= min_thresh:
            min_val = i
            break
    
    for i in range(255, -1, -1):
        if cumsum[i] <= max_thresh:
            max_val = i
            break
    
    # Yeni bir görüntü oluştur
    spread = np.zeros_like(channel, dtype=np.float32)
    
    # Kontrast yayma formülü: yeni_piksel = (piksel - min_val) * 255 / (max_val - min_val)
    if min_val < max_val:
        spread = np.clip((channel.astype(np.float32) - min_val) * 255 / (max_val - min_val), 0, 255)
    
    # Değerleri uint8'e dönüştür
    return spread.astype(np.uint8)