"""
Teste simples: verificar valores de pixels após preprocessing.
"""
import cv2
import numpy as np
from pathlib import Path

def test_preprocessing(image_path: str):
    """Verifica valores de pixels após cada etapa."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))
    
    # 1. Equalização
    equalized = cv2.equalizeHist(img_resized)
    
    # 2. Bilateral
    bilateral = cv2.bilateralFilter(equalized, 9, 75, 75)
    
    # 3. Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
    
    # 4. Otsu
    _, binary = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    print(f"Imagem: {Path(image_path).name}")
    print(f"Original: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
    print(f"Resized: min={img_resized.min()}, max={img_resized.max()}, mean={img_resized.mean():.1f}")
    print(f"Equalized: min={equalized.min()}, max={equalized.max()}, mean={equalized.mean():.1f}")
    print(f"Bilateral: min={bilateral.min()}, max={bilateral.max()}, mean={bilateral.mean():.1f}")
    print(f"Morph: min={morph.min()}, max={morph.max()}, mean={morph.mean():.1f}")
    print(f"Binary: min={binary.min()}, max={binary.max()}, mean={binary.mean():.1f}")
    print(f"  Pixels brancos (255): {np.sum(binary == 255)}")
    print(f"  Pixels pretos (0): {np.sum(binary == 0)}")
    
    # Dividir em grid 3x2 e verificar cada célula
    h, w = binary.shape
    cell_h = h // 3
    cell_w = w // 2
    
    print(f"\nGrid 3x2 (cada célula {cell_h}x{cell_w}):")
    for row in range(3):
        for col in range(2):
            y0 = row * cell_h
            y1 = (row + 1) * cell_h
            x0 = col * cell_w
            x1 = (col + 1) * cell_w
            
            cell = binary[y0:y1, x0:x1]
            white_pixels = np.sum(cell == 255)
            total_pixels = cell.size
            white_ratio = white_pixels / total_pixels
            
            dot_num = {(0,0):1, (1,0):2, (2,0):3, (0,1):4, (1,1):5, (2,1):6}[(row, col)]
            print(f"  Dot {dot_num} (row={row}, col={col}): {white_pixels}/{total_pixels} brancos = {white_ratio:.2%}")

if __name__ == "__main__":
    # Testar letra A (dot 1 apenas)
    test_preprocessing("dataset/a1.JPG0dim.jpg")
    print("\n" + "="*70 + "\n")
    # Testar letra D (dots 1, 4, 5)
    test_preprocessing("dataset/d1.JPG0dim.jpg")
