"""
Debug script para visualizar processamento Braille.
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Importar funções do run_no_ml
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src/classification_no_ml"))
from run_no_ml import (
    preprocess_image,
    image_to_braille_dots,
    dots_to_matrix,
    BRAILLE_DOTS_MAP,
    IMG_SIZE
)

def debug_image(image_path: str):
    """Processa e visualiza os passos de detecção."""
    # Carregar imagem
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Erro ao carregar: {image_path}")
        return
    
    # Redimensionar
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Preprocessar
    binary = preprocess_image(img_resized)
    
    # Detectar dots
    dots = image_to_braille_dots(img_resized)
    matrix = dots_to_matrix(dots)
    
    # Pegar letra do nome do arquivo
    true_label = Path(image_path).name[0].upper()
    expected_dots = BRAILLE_DOTS_MAP.get(true_label, [])
    expected_matrix = dots_to_matrix(expected_dots)
    
    # Visualizar
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title(f'Original\nLabel: {true_label}')
    axes[0, 0].axis('off')
    
    # Redimensionada
    axes[0, 1].imshow(img_resized, cmap='gray')
    axes[0, 1].set_title(f'Resized {IMG_SIZE}')
    axes[0, 1].axis('off')
    
    # Binária (preprocessada)
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('Preprocessed (Binary)')
    axes[0, 2].axis('off')
    
    # Matrix esperada
    axes[1, 0].imshow(expected_matrix, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Expected Matrix\nDots: {expected_dots}')
    axes[1, 0].axis('off')
    
    # Matrix detectada
    axes[1, 1].imshow(matrix, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Detected Matrix\nDots: {sorted(dots)}')
    axes[1, 1].axis('off')
    
    # Comparação
    match = np.array_equal(matrix, expected_matrix)
    diff = np.abs(matrix - expected_matrix)
    axes[1, 2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Difference\nMatch: {match}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    results_dir = PROJECT_ROOT / "results" / "no_ml"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"debug_{true_label.lower()}.png"
    plt.savefig(str(output_path), dpi=150)
    print(f"✓ Salvo em: {output_path}")
    print(f"  Esperado: {expected_dots}")
    print(f"  Detectado: {sorted(dots)}")
    print(f"  Match: {match}\n")
    plt.close()


if __name__ == "__main__":
    # Testar algumas letras
    dataset_dir = Path("dataset")
    
    test_letters = ['a', 'b', 'c', 'd', 'e']
    for letter in test_letters:
        # Pegar primeira imagem da letra
        files = list(dataset_dir.glob(f"{letter}1.JPG0dim.jpg"))
        if files:
            print(f"\n{'='*60}")
            print(f"Testando letra: {letter.upper()}")
            print(f"{'='*60}")
            debug_image(str(files[0]))
