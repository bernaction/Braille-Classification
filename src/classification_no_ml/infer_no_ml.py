# src/classification_no_ml/infer_no_ml.py
"""Inferência sem ML para webcam.
Reusa lógica de matriz + histograma acumulativo do classificador simples.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2

from src.preprocessing.pipeline import preprocess_for_braille

BRAILLE_DOTS_MAP: Dict[str, List[int]] = {
    "A": [1], "B": [1,2], "C": [1,4], "D": [1,4,5], "E": [1,5],
    "F": [1,2,4], "G": [1,2,4,5], "H": [1,2,5], "I": [2,4], "J": [2,4,5],
    "K": [1,3], "L": [1,2,3], "M": [1,3,4], "N": [1,3,4,5], "O": [1,3,5],
    "P": [1,2,3,4], "Q": [1,2,3,4,5], "R": [1,2,3,5], "S": [2,3,4], "T": [2,3,4,5],
    "U": [1,3,6], "V": [1,2,3,6], "W": [2,4,5,6], "X": [1,3,4,6], "Y": [1,3,4,5,6], "Z": [1,3,5,6]
}

_mapping = {1:(0,0),2:(1,0),3:(2,0),4:(0,1),5:(1,1),6:(2,1)}

BRAILLE_MATRIX_MAP: Dict[str, np.ndarray] = {}
for ch,dots in BRAILLE_DOTS_MAP.items():
    mat = np.zeros((3,2), dtype=int)
    for d in dots:
        r,c = _mapping[d]
        mat[r,c] = 1
    BRAILLE_MATRIX_MAP[ch] = mat


def _cell_dot_score(cell: np.ndarray) -> float:
    hist = np.zeros(256, dtype=np.int32)
    flat = cell.ravel()
    for v in flat:
        hist[v] += 1
    cum = np.cumsum(hist)
    below_200 = cum[200]
    return 1.0 - (below_200 / len(flat))


def _detect_dots(bin_img: np.ndarray) -> Tuple[List[int], Dict[int,float]]:
    h, w = bin_img.shape
    cell_h = h // 3
    cell_w = w // 2
    mapping = {(0,0):1,(1,0):2,(2,0):3,(0,1):4,(1,1):5,(2,1):6}
    scores = {}
    for r in range(3):
        for c in range(2):
            cell = bin_img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            score = _cell_dot_score(cell)
            dot = mapping[(r,c)]
            scores[dot] = score
    mean_score = np.mean(list(scores.values())) if scores else 0.0
    thr = max(0.15, mean_score * 0.85)
    active = [d for d,s in scores.items() if s >= thr]
    if not active and scores:
        # fallback pega melhor
        best = max(scores.items(), key=lambda x: x[1])[0]
        active = [best]
    return sorted(active), scores


def classify_frame(frame_bgr: np.ndarray, resize: Tuple[int,int]=(128,128)) -> Tuple[str, float, List[int]]:
    gray = preprocess_for_braille(frame_bgr, resize=resize, return_binary=True, adaptive=False, use_roi=False)
    dots, scores = _detect_dots(gray)
    mat = np.zeros((3,2), dtype=int)
    for d in dots:
        r,c = _mapping[d]
        mat[r,c] = 1
    pred = "?"
    for ch, ref in BRAILLE_MATRIX_MAP.items():
        if np.array_equal(mat, ref):
            pred = ch
            break
    # confiança simples: média dos scores dos dots ativos (ou média geral se vazio)
    conf = float(np.mean([scores[d] for d in dots])) if dots else float(np.mean(list(scores.values()) or [0.0]))
    return pred, conf, dots

if __name__ == "__main__":
    # teste rápido com primeira imagem do dataset (se existir)
    sample = next((p for p in Path("dataset").rglob("*.png")), None)
    if sample:
        img = cv2.imread(str(sample))
        pred, conf, dots = classify_frame(img)
        print(f"Teste rápido: pred={pred} conf={conf:.3f} dots={dots}")
    else:
        print("Nenhuma imagem encontrada para teste.")
