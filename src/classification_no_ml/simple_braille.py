# src/classification_no_ml/simple_braille.py
"""Classificador Braille FROM SCRATCH usando apenas:
- Pré-processamento padronizado
- Histograma acumulativo / proporção de pixels brancos por célula
- Lógica de matriz 3x2
Cumpre o item B (sem aprendizado de máquina).
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

import sys
from pathlib import Path as _PathForSys
# Garante que raiz do projeto esteja no sys.path para imports 'src.*'
ROOT_DIR = _PathForSys(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.pipeline import preprocess_for_braille

DATASET_DIR = Path("dataset")
TEST_SIZE = 0.2
RANDOM_STATE = 42
IMG_SIZE = (128, 128)
RESULTS_DIR = Path("results/no_ml/simple")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Mapeamento Braille
BRAILLE_DOTS_MAP: Dict[str, List[int]] = {
    "A": [1], "B": [1,2], "C": [1,4], "D": [1,4,5], "E": [1,5],
    "F": [1,2,4], "G": [1,2,4,5], "H": [1,2,5], "I": [2,4], "J": [2,4,5],
    "K": [1,3], "L": [1,2,3], "M": [1,3,4], "N": [1,3,4,5], "O": [1,3,5],
    "P": [1,2,3,4], "Q": [1,2,3,4,5], "R": [1,2,3,5], "S": [2,3,4], "T": [2,3,4,5],
    "U": [1,3,6], "V": [1,2,3,6], "W": [2,4,5,6], "X": [1,3,4,6], "Y": [1,3,4,5,6], "Z": [1,3,5,6]
}

def dots_to_matrix(dots: List[int]) -> np.ndarray:
    mat = np.zeros((3,2), dtype=int)
    mapping = {1:(0,0),2:(1,0),3:(2,0),4:(0,1),5:(1,1),6:(2,1)}
    for d in dots:
        if d in mapping:
            r,c = mapping[d]
            mat[r,c] = 1
    return mat

BRAILLE_MATRIX_MAP: Dict[str, np.ndarray] = {ch: dots_to_matrix(d) for ch,d in BRAILLE_DOTS_MAP.items()}

def infer_label_from_path(path: Path) -> str:
    for ch in path.name:
        if ch.isalpha():
            return ch.upper()
    return "?"


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths: List[Path] = []
    for ext in ("*.png","*.jpg","*.jpeg"):
        paths.extend(DATASET_DIR.rglob(ext))
    X = []
    y = []
    for p in paths:
        raw = cv2.imread(str(p))
        if raw is None:
            continue
        proc = preprocess_for_braille(raw, resize=IMG_SIZE, return_binary=True, adaptive=False, use_roi=False)
        label = infer_label_from_path(p)
        X.append(proc)
        y.append(label)
    X = np.array(X, dtype=np.uint8)
    y = np.array(y)
    if len(set(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
    print(f"[dataset] Total: {len(X)} | Classes: {len(set(y))}")
    return X_train, X_test, y_train, y_test


def cell_white_ratio(cell: np.ndarray) -> float:
    return float(np.sum(cell >= 200) / cell.size)


def cell_dot_score(cell: np.ndarray) -> float:
    # Histograma acumulativo FROM SCRATCH
    hist = np.zeros(256, dtype=np.int32)
    flat = cell.ravel()
    for v in flat:
        hist[v] += 1
    cum = np.cumsum(hist)
    total = len(flat)
    # Proporção de pixels >= 200
    below_200 = cum[200]
    return 1.0 - (below_200 / total)


def detect_dots(image: np.ndarray) -> List[int]:
    h, w = image.shape
    cell_h = h // 3
    cell_w = w // 2
    mapping = { (0,0):1,(1,0):2,(2,0):3,(0,1):4,(1,1):5,(2,1):6 }
    dots: List[int] = []
    ratios: List[Tuple[int,float]] = []
    for r in range(3):
        for c in range(2):
            cell = image[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            score = cell_dot_score(cell)  # usa cumulativo
            ratios.append((mapping[(r,c)], score))
    # Threshold simples: escolhe aqueles com score > média - pequena margem
    scores = [s for _,s in ratios]
    mean_score = float(np.mean(scores))
    thr = max(0.15, mean_score * 0.85)  # adaptativo simples
    for d,s in ratios:
        if s >= thr:
            dots.append(d)
    # Garantias mínimas
    if len(dots) == 0:
        # pega top 1
        top_dot = sorted(ratios, key=lambda x: x[1], reverse=True)[0][0]
        dots = [top_dot]
    return sorted(dots)


def classify_image(image: np.ndarray) -> str:
    dots = detect_dots(image)
    mat = dots_to_matrix(dots)
    for label, ref in BRAILLE_MATRIX_MAP.items():
        if np.array_equal(mat, ref):
            return label
    return "?"


def evaluate():
    X_train, X_test, y_train, y_test = load_dataset()
    print("[eval] Classificando teste...")
    y_pred = [classify_image(img) for img in X_test]
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia (sem ML simples): {acc*100:.2f}%")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    print("Matriz de confusão:\n", cm)
    print(classification_report(y_test, y_pred, labels=sorted(set(y_test))))

    # Salvar matriz de confusão como imagem
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de Confusão - Sem ML (Simples)")
    classes = sorted(set(y_test))
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=6)
    plt.tight_layout()
    out_img = RESULTS_DIR / "confusion_matrix_simple.png"
    plt.savefig(out_img, dpi=150)
    plt.close(fig)
    print(f"[eval] Confusion matrix salva em {out_img}")

    # Salvar resumo em texto
    summary_path = RESULTS_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Acurácia: {acc*100:.2f}%\n")
        f.write("Classes: " + ",".join(sorted(set(y_test))) + "\n")
    print(f"[eval] Resumo salvo em {summary_path}")


if __name__ == "__main__":
    evaluate()
