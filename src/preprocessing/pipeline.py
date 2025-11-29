# src/preprocessing/pipeline.py

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

# Pipeline padronizado para processamento de caracteres Braille
# Etapas (A):
# 1. Garantir grayscale
# 2. Suavização (GaussianBlur)
# 3. Equalização de histograma
# 4. Filtro morfológico (fechamento) para realçar pontos
# 5. Limiarização adaptativa ou Otsu (binarização)
# 6. (Opcional) Detecção simples de ROI via maior contorno
# 7. Retorno imagem binária ou normalizada dependendo do uso


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def gaussian(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def equalize(image: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(image)


def morph_close(image: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def binarize(image: np.ndarray, adaptive: bool = False, invert: bool = True) -> np.ndarray:
    if adaptive:
        # Adaptive mean threshold
        th = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
            15,
            5,
        )
        return th
    # Otsu
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, th = cv2.threshold(image, 0, 255, flag + cv2.THRESH_OTSU)
    return th


def detect_roi(image: np.ndarray) -> np.ndarray:
    # ROI simples: maior contorno em imagem binária para recorte aproximado
    bin_img = binarize(image, adaptive=False, invert=True)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    # Margem pequena
    pad = 4
    y0 = max(y - pad, 0)
    x0 = max(x - pad, 0)
    y1 = min(y + h + pad, image.shape[0])
    x1 = min(x + w + pad, image.shape[1])
    cropped = image[y0:y1, x0:x1]
    # Evita recorte minúsculo acidental
    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return image
    return cropped


def preprocess_for_braille(
    image: np.ndarray,
    resize: tuple[int, int] | None = None,
    return_binary: bool = False,
    adaptive: bool = False,
    use_roi: bool = False,
) -> np.ndarray:
    """Pipeline completa utilizada por ambos métodos.
    return_binary=True devolve imagem binária (para lógica sem ML), senão grayscale equalizada.
    """
    img = ensure_gray(image)
    img = gaussian(img, 5)
    img = equalize(img)
    img = morph_close(img, 3)
    if use_roi:
        img = detect_roi(img)
    if return_binary:
        img = binarize(img, adaptive=adaptive, invert=True)
    if resize is not None:
        img = cv2.resize(img, resize)
    return img


if __name__ == "__main__":
    # Demonstração rápida
    sample_path = next((p for p in Path("dataset").rglob("*.png")), None)
    if sample_path is None:
        print("[pipeline] Nenhuma imagem encontrada para demonstração.")
    else:
        raw = cv2.imread(str(sample_path))
        proc_bin = preprocess_for_braille(raw, resize=(128,128), return_binary=True)
        proc_gray = preprocess_for_braille(raw, resize=(128,128), return_binary=False)
        Path("results/preprocessing").mkdir(parents=True, exist_ok=True)
        cv2.imwrite("results/preprocessing/demo_binary.png", proc_bin)
        cv2.imwrite("results/preprocessing/demo_gray.png", proc_gray)
        print("[pipeline] Imagens de demonstração salvas em results/preprocessing/")
