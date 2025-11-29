#!/usr/bin/env python3
"""Main orchestrator for Braille Classification project.
Provides CLI to run:
  - Histogram analyses
  - No-ML baseline evaluation
  - CNN training
"""
from pathlib import Path
import argparse
import sys
import cv2
import numpy as np


from classification_no_ml.run_no_ml import evaluate_no_ml, load_braille_dataset  # type: ignore
from classification_cnn.train_cnn import main as train_cnn_main  # type: ignore
from preprocessing.histogram_utils import (
    plot_cumulative_histogram,
    plot_letter_histograms,
    compute_dot_cell_cdf,
    summarize_cdf_contrast,
)

def run_histograms(sample_per_letter: int = 5):
    print('[MAIN] Executando análise de histogramas...')
    X_train, X_test, y_train, y_test = load_braille_dataset()
    # Agrupar amostras por letra (treino)
    letter_samples = {}
    for img, label in zip(X_train, y_train):
        letter_samples.setdefault(label, []).append(img)
    # Reduzir para n amostras por letra
    for k in letter_samples:
        letter_samples[k] = letter_samples[k][:sample_per_letter]
    out_dir = PROJECT_ROOT / 'results' / 'histograms'
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_letter_histograms(letter_samples, save_dir=out_dir, show=False)
    # Gerar cumulativo de uma amostra por letra
    for letter, imgs in letter_samples.items():
        if not imgs:
            continue
        plot_cumulative_histogram(imgs[0], title=f'Cumulativo {letter}', show=False,
                                   save_path=out_dir / f'cumulative_{letter}.png')
    # Análise de CDF por célula (primeira letra)
    ref_letter = next(iter(letter_samples))
    ref_img = letter_samples[ref_letter][0]
    cdf_cells = compute_dot_cell_cdf(ref_img)
    spreads = [[summarize_cdf_contrast(cdf_cells[r][c]) for c in range(2)] for r in range(3)]
    np.save(out_dir / 'cell_spreads.npy', np.array(spreads))
    print(f'[MAIN] Spreads (diferença índice p90-p10) por célula da letra {ref_letter}: {spreads}')
    print(f'[MAIN] Arquivos salvos em {out_dir}')


def parse_args():
    p = argparse.ArgumentParser(description='Braille Classification Project Orchestrator')
    p.add_argument('--hist', action='store_true', help='Executa análise de histogramas')
    p.add_argument('--no-ml', action='store_true', help='Roda classificação sem ML')
    p.add_argument('--cnn', action='store_true', help='Treina CNN')
    p.add_argument('--samples-per-letter', type=int, default=5, help='Amostras por letra para histograma')
    return p.parse_args()


def main():
    args = parse_args()
    if not any([args.hist, args.no_ml, args.cnn]):
        print('Nada para executar. Use --hist / --no-ml / --cnn.')
        return
    if args.hist:
        run_histograms(sample_per_letter=args.samples_per_letter)
    if args.no_ml:
        evaluate_no_ml()
    if args.cnn:
        train_cnn_main()

if __name__ == '__main__':
    main()
