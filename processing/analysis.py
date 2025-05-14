import numpy as np
import cv2

def center_of_mass(binary_image):
    # Beyaz piksellerin ağırlık merkezi
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    binary = (binary_image > 0).astype(np.uint8)
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return None
    y, x = coords.mean(axis=0)
    return int(x), int(y)

def mark_center_of_mass(image, center):
    # Ağırlık merkezini kırmızı bir daire ile işaretle
    marked = image.copy()
    if len(marked.shape) == 2:
        marked = cv2.cvtColor(marked, cv2.COLOR_GRAY2BGR)
    if center is not None:
        cv2.circle(marked, center, 8, (0, 0, 255), -1)
    return marked

def zhang_suen_thinning(binary_image):
    # Gri veya renkli ise binary'ye çevir
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    
    # Binary görüntüyü hazırla
    _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    
    # OpenCV'nin iskelet çıkarma fonksiyonu
    skeleton = cv2.ximgproc.thinning(binary)
    
    return skeleton