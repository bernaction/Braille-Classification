# src/preprocessing/histogram_utils.py

from pathlib import Path
from typing import Iterable, Dict, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats


def compute_histogram(image: np.ndarray, nbins: int = 256) -> np.ndarray:
    """
    Calcula o histograma de níveis de cinza manualmente (from scratch).
    image: imagem em escala de cinza (uint8)
    """
    if image.ndim != 2:
        raise ValueError("compute_histogram: a imagem deve ser grayscale (2D).")

    hist = np.zeros(nbins, dtype=np.int64)
    flat = image.ravel()

    for value in flat:
        hist[int(value)] += 1

    return hist


def compute_cumulative_histogram(hist: np.ndarray) -> np.ndarray:
    """
    Calcula o histograma acumulativo (CDF) a partir de um histograma.
    """
    return np.cumsum(hist)


def plot_cumulative_histogram(
    image: np.ndarray,
    title: str = "Cumulative Histogram",
    save_path: str | Path | None = None,
    show: bool = True,
    use_scipy: bool = True,
):
    """Plota histograma e cumulativo.
    Se use_scipy=True usa stats.cumfreq para ilustrar método do professor.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = compute_histogram(image)
    cum_hist = compute_cumulative_histogram(hist)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(hist, color="green")
    ax[0].set_title("Histograma (from scratch)")
    ax[0].set_xlabel("Intensidade (0-255)")
    ax[0].set_ylabel("Frequência")

    if use_scipy:
        # Normaliza para [0,1] e usa cumfreq com nbins reduzidos para demonstrar
        flat = image.ravel().astype(float)
        res = stats.cumfreq(flat, numbins=32)
        x_vals = res.lowerlimit + np.linspace(
            0, res.binsize * res.cumcount.size, res.cumcount.size
        )
        ax[1].bar(x_vals, res.cumcount, width=res.binsize, color="blue")
        ax[1].set_title("Histograma Acumulativo (scipy.cumfreq)")
    else:
        ax[1].plot(cum_hist, color="blue")
        ax[1].set_title("Histograma Acumulativo (from scratch)")
    ax[1].set_xlabel("Intensidade")
    ax[1].set_ylabel("Frequência acumulada")

    fig.suptitle(title)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_letter_histograms(
    samples: Dict[str, Iterable[np.ndarray]],
    nbins: int = 256,
    save_dir: str | Path | None = None,
    show: bool = False,
):
    """Gera histogramas médios e cumulativos por letra.
    samples: dict letra -> iterável de imagens grayscale.
    """
    rows = int(np.ceil(len(samples) / 6))
    fig, axes = plt.subplots(rows, 6, figsize=(18, 3 * rows))
    axes = axes.flatten()
    for idx, (letter, imgs) in enumerate(sorted(samples.items())):
        imgs_list = list(imgs)
        if not imgs_list:
            continue
        hists = []
        for im in imgs_list[:10]:  # limita até 10 amostras por letra para média
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            hists.append(compute_histogram(im, nbins))
        mean_hist = np.mean(hists, axis=0)
        axes[idx].plot(mean_hist, color="black")
        axes[idx].set_title(letter)
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    plt.tight_layout()
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "letters_mean_histograms.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"[hist] Salvo: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_dot_cell_cdf(image: np.ndarray, grid: Tuple[int, int] = (3, 2)) -> np.ndarray:
    """Divide a imagem em grid (3x2 Braille) e retorna matriz de CDF normalizada por célula."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    rows, cols = grid
    cell_h = h // rows
    cell_w = w // cols
    cdfs = []
    for r in range(rows):
        row_cdfs = []
        for c in range(cols):
            cell = image[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            hist = compute_histogram(cell)
            cdf = compute_cumulative_histogram(hist).astype(float)
            cdf /= cdf[-1] if cdf[-1] != 0 else 1.0
            row_cdfs.append(cdf)
        cdfs.append(row_cdfs)
    return np.array(cdfs, dtype=object)  # matriz 3x2 de arrays 1D


def summarize_cdf_contrast(cdf: np.ndarray) -> float:
    """Resumo simples: diferença entre percentis 90% e 10% como medida de spread."""
    # cdf é array 1D normalizado
    # encontrar índice onde cdf >= p
    def idx(p: float) -> int:
        return int(np.argmax(cdf >= p))
    return float(idx(0.9) - idx(0.1))

if __name__ == "__main__":
    # Demonstração rápida usando ruído sintético
    rng = np.random.default_rng(42)
    synthetic = (rng.normal(128, 40, size=(128,128))).clip(0,255).astype(np.uint8)
    plot_cumulative_histogram(synthetic, title="Demo Synthetic", show=False, save_path="results/histograms/demo_synthetic.png")
    print("Histograma demonstrativo gerado em results/histograms/demo_synthetic.png")
