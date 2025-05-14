import numpy as np
import cv2

def manual_threshold(image, threshold=127):
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = np.zeros_like(image)
    binary[image > threshold] = 255
    return binary

def otsu_threshold(image):
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def kapur_threshold(image):
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum()
    cumsum = np.cumsum(hist)
    entropy_b = np.zeros(256)
    entropy_f = np.zeros(256)
    for t in range(256):
        # Background
        if cumsum[t] > 0:
            p_b = hist[:t+1] / cumsum[t]
            entropy_b[t] = -np.sum(p_b[p_b > 0] * np.log(p_b[p_b > 0]))
        # Foreground
        if cumsum[t] < 1:
            p_f = hist[t+1:] / (1 - cumsum[t])
            entropy_f[t] = -np.sum(p_f[p_f > 0] * np.log(p_f[p_f > 0]))
    kapur = entropy_b + entropy_f
    thresh = np.argmax(kapur)
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return binary

def local_threshold(image, block_size=16, c=5):
    """
    Elle yazılmış yerel eşikleme (local thresholding) algoritması.
    Bu fonksiyon görüntüyü küçük bloklara böler ve her bir blok için ayrı
    eşik değerleri hesaplayarak adaptif eşikleme yapar (shading correction).
    
    Parametreler:
    - image: Eşiklenecek görüntü
    - block_size: Yerel bölge boyutu (varsayılan: 16x16 piksel)
    - c: Eşik değerinden çıkarılacak sabit (düşük değerler daha fazla beyaz piksel)
    
    Dönüş:
    - binary: İkili (binary) görüntü 
    """
    # Gri tonlamalı görüntüye çevir
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Görüntü boyutları
    height, width = image.shape
    
    # Çıkış görüntüsü
    binary = np.zeros_like(image)
    
    # Görüntüyü bloklara böl
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Blok sınırlarını belirle
            block_end_y = min(y + block_size, height)
            block_end_x = min(x + block_size, width)
            
            # Bloğu al
            block = image[y:block_end_y, x:block_end_x]
            
            # Blok için eşik değeri hesapla (ortalama - c)
            threshold = np.mean(block) - c
            
            # Eşikleme uygula
            mask = block > threshold
            binary[y:block_end_y, x:block_end_x][mask] = 255
    
    return binary

def adaptive_local_threshold(image, window_size=51, c=10):
    """
    Gelişmiş adaptif yerel eşikleme.
    Her piksel için etrafındaki bir pencere alınır ve piksel değeri
    yerel pencere ortalamasının altındaysa 0, üstündeyse 255 olarak ayarlanır.
    
    Parametreler:
    - image: Eşiklenecek görüntü
    - window_size: Yerel pencere boyutu (tek sayı olmalı)
    - c: Ortalamadan çıkarılacak sabit
    
    Dönüş:
    - binary: İkili (binary) görüntü
    """
    # Gri tonlamalı görüntüye çevir
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Görüntü boyutları
    height, width = image.shape
    
    # Çıkış görüntüsü
    binary = np.zeros_like(image)
    
    # Kenar padding miktarı
    pad = window_size // 2
    
    # Kenarları aynalayarak görüntüyü genişlet
    padded = np.pad(image, pad, mode='reflect')
    
    # Her piksel için
    for i in range(height):
        for j in range(width):
            # Yerel pencereyi al
            window = padded[i:i+window_size, j:j+window_size]
            
            # Yerel ortalama hesapla
            local_mean = np.mean(window)
            
            # Eşikleme uygula
            if image[i, j] > local_mean - c:
                binary[i, j] = 255
    
    return binary 