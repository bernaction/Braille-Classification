# src/classification_no_ml/run_no_ml.py

from pathlib import Path
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
from functools import partial

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


IMG_SIZE = (128, 128)  # Aumentado para melhor detecção
DATASET_DIR = Path("dataset")
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD_VALUE = 30  # Threshold para detecção de dots


# ========= DICIONÁRIO BRAILLE (A-Z) EM DOTS =========
# dots são numerados assim:
# 1 4
# 2 5
# 3 6
BRAILLE_DOTS_MAP: Dict[str, List[int]] = {
    "A": [1],
    "B": [1, 2],
    "C": [1, 4],
    "D": [1, 4, 5],
    "E": [1, 5],
    "F": [1, 2, 4],
    "G": [1, 2, 4, 5],
    "H": [1, 2, 5],
    "I": [2, 4],
    "J": [2, 4, 5],
    "K": [1, 3],
    "L": [1, 2, 3],
    "M": [1, 3, 4],
    "N": [1, 3, 4, 5],
    "O": [1, 3, 5],
    "P": [1, 2, 3, 4],
    "Q": [1, 2, 3, 4, 5],
    "R": [1, 2, 3, 5],
    "S": [2, 3, 4],
    "T": [2, 3, 4, 5],
    "U": [1, 3, 6],
    "V": [1, 2, 3, 6],
    "W": [2, 4, 5, 6],
    "X": [1, 3, 4, 6],
    "Y": [1, 3, 4, 5, 6],
    "Z": [1, 3, 5, 6],
}


def dots_to_matrix(dots: List[int]) -> np.ndarray:
    """
    Converte lista de dots [1..6] em matriz 3x2 binária.
    """
    mat = np.zeros((3, 2), dtype=int)
    mapping = {
        1: (0, 0),
        2: (1, 0),
        3: (2, 0),
        4: (0, 1),
        5: (1, 1),
        6: (2, 1),
    }
    for d in dots:
        r, c = mapping[d]
        mat[r, c] = 1
    return mat


BRAILLE_MATRIX_MAP: Dict[str, np.ndarray] = {
    ch: dots_to_matrix(dots) for ch, dots in BRAILLE_DOTS_MAP.items()
}


def infer_label_from_path(path: Path) -> str:
    """Extrai a primeira letra alfabética do nome do arquivo.
    Ignora lógica de pastas (dataset plano). FROM SCRATCH simples.
    """
    fname = path.name
    for ch in fname:
        if ch.isalpha():
            return ch.upper()
    return "?"


def load_braille_dataset(
    dataset_dir: Path = DATASET_DIR,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega imagens do dataset e faz split treino/teste.
    Aceita imagens .png, .jpg, .jpeg.
    """
    image_paths: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(dataset_dir.rglob(ext))

    X = []
    y = []

    label_counts: Dict[str, int] = {}
    sample_map: Dict[str, str] = {}
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, img_size)
        label = infer_label_from_path(path)

        X.append(img)
        y.append(label)
        label_counts[label] = label_counts.get(label, 0) + 1
        if label not in sample_map:
            sample_map[label] = path.name

    # DEBUG: imprimir distribuição real de labels antes do split
    print("\n[DEBUG] Distribuição de labels carregadas:")
    for k in sorted(label_counts.keys()):
        print(f"  - {k}: {label_counts[k]}")
    print(f"[DEBUG] Total labels distintos: {len(label_counts)}")
    print("[DEBUG] Exemplo por label:")
    for k in sorted(sample_map.keys()):
        print(f"  > {k}: {sample_map[k]}")

    X = np.array(X, dtype=np.uint8)
    y = np.array(y)

    if len(set(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    else:
        # Sem estratificação possível
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

    return X_train, X_test, y_train, y_test


def preprocess_image(image: np.ndarray, invert: bool = True) -> np.ndarray:
    """
    Aplica pré-processamento avançado na imagem:
    1. Equalização de histograma
    2. Filtro bilateral para suavizar preservando bordas
    3. Filtro morfológico para realçar pontos
    4. Binarização adaptativa com Otsu
    """
    # 1. Equalização de histograma para melhorar contraste
    equalized = cv2.equalizeHist(image)
    
    # 2. Filtro bilateral: suaviza mas preserva bordas
    bilateral = cv2.bilateralFilter(equalized, 9, 75, 75)
    
    # 3. Morfologia: dilate + erode para realçar pontos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
    
    # 4. Binarização com Otsu (inverte para dots brancos)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(
        morph, 0, 255, flag + cv2.THRESH_OTSU
    )
    
    return binary


def compute_cumulative_histogram(cell: np.ndarray) -> float:
    """
    Calcula histograma acumulativo da célula.
    Retorna score normalizado indicando presença de dot.
    
    Abordagem FROM SCRATCH:
    1. Calcular histograma de intensidades [0-255]
    2. Acumular valores do histograma
    3. Normalizar pelo total de pixels
    4. Usar threshold para decidir presença de dot
    """
    # 1. Histograma: contar pixels em cada intensidade [0-255]
    hist = cv2.calcHist([cell], [0], None, [256], [0, 256]).flatten()
    
    # 2. Histograma acumulativo: soma acumulada
    cum_hist = np.cumsum(hist)
    
    # 3. Normalizar pelo total de pixels
    total_pixels = cell.size
    cum_hist_norm = cum_hist / total_pixels
    
    # 4. Score: proporção de pixels com intensidade > 200 (brancos)
    # Em imagem binária preprocessada, dots brancos têm intensidade 255
    # Usamos cum_hist_norm[200] para contar pixels abaixo de 200
    # Então 1.0 - cum_hist_norm[200] = proporção de pixels >= 200 (brancos)
    score = 1.0 - cum_hist_norm[200]
    
    return score


def image_to_braille_dots(image: np.ndarray) -> List[int]:
    """
    Nova estratégia:
    1. Pré-processamento SEM inversão (dots claros ou escuros?)
    2. Calcular proporção de pixels brancos (>=200) em cada célula
    3. Aplicar K-Means (k=2) sobre proporções para separar ativos/inativos
    4. Se cluster alto tiver proporção média <0.35, usar fallback histograma acumulativo.
    5. Garantir quantidade plausível (1..6). Se tudo ativado, aplica threshold dinâmico.
    FROM SCRATCH: proporção + histograma cumulativo
    """
    # Tentativa 0: detectar círculos (HoughCircles) diretamente na imagem equalizada
    eq = cv2.equalizeHist(image)
    circles = cv2.HoughCircles(
        eq,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=15,
        minRadius=6,
        maxRadius=26,
    )
    cell_to_dot = {
        (0, 0): 1,
        (1, 0): 2,
        (2, 0): 3,
        (0, 1): 4,
        (1, 1): 5,
        (2, 1): 6,
    }
    h, w = eq.shape
    cell_h = h // 3
    cell_w = w // 2
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        dots_hough: List[int] = []
        for (x, y, r) in circles:
            if r < 6 or r > 30:
                continue
            row = int(y // cell_h)
            col = int(x // cell_w)
            if 0 <= row < 3 and 0 <= col < 2:
                d = cell_to_dot[(row, col)]
                if d not in dots_hough:
                    dots_hough.append(d)
        if 0 < len(dots_hough) <= 6:
            return sorted(dots_hough)

    # Usar binarização não invertida para proporções (Tentativa 1)
    binary_no_inv = preprocess_image(image, invert=False)

    h, w = binary_no_inv.shape
    cell_h = h // 3
    cell_w = w // 2

    ratios: List[float] = []
    cells: List[Tuple[int,int,np.ndarray]] = []
    for row in range(3):
        for col in range(2):
            y0 = row * cell_h
            y1 = (row + 1) * cell_h
            x0 = col * cell_w
            x1 = (col + 1) * cell_w
            cell = binary_no_inv[y0:y1, x0:x1]
            white = np.sum(cell >= 200)
            ratio = white / cell.size
            ratios.append(ratio)
            cells.append((row, col, cell))

    ratios_np = np.array(ratios).reshape(-1, 1)
    try:
        km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init='auto')
        km.fit(ratios_np)
        labels_km = km.labels_
        cluster_means = [ratios_np[labels_km == i].mean() for i in range(2)]
        high_cluster = int(np.argmax(cluster_means))
        high_mean = cluster_means[high_cluster]
        active_indices = [i for i,l in enumerate(labels_km) if l == high_cluster]
    except Exception:
        # Fallback simples: threshold fixo se KMeans falhar
        active_indices = [i for i,r in enumerate(ratios) if r > 0.50]
        high_mean = np.mean([ratios[i] for i in active_indices]) if active_indices else 0.0

    active_dots: List[int] = []
    for idx in active_indices:
        row, col, _ = cells[idx]
        active_dots.append(cell_to_dot[(row, col)])

    # Se cluster alto não for convincente, usar histograma acumulativo por célula
    if high_mean < 0.35 or len(active_dots) == 0:
        active_dots = []
        binary_inv = preprocess_image(image, invert=True)
        for row in range(3):
            for col in range(2):
                y0 = row * cell_h
                y1 = (row + 1) * cell_h
                x0 = col * cell_w
                x1 = (col + 1) * cell_w
                cell = binary_inv[y0:y1, x0:x1]
                score = compute_cumulative_histogram(cell)
                if score > 0.45:  # Mais restritivo
                    active_dots.append(cell_to_dot[(row, col)])

    # Correção caso todos os 6 apareçam: aplica corte dinâmico
    if len(active_dots) > 4:
        # Ordena pares (dot, ratio) e mantém apenas aqueles com ratio dentro de topo - queda
        dot_ratios = [(cell_to_dot[(row, col)], ratios[i]) for i,(row,col,_) in enumerate(cells)]
        dot_ratios.sort(key=lambda x: x[1], reverse=True)
        # Mantém sempre top 1, depois mantém enquanto diferença < 0.12
        filtered = [dot_ratios[0][0]]
        base = dot_ratios[0][1]
        for d,r in dot_ratios[1:]:
            if base - r < 0.12:
                filtered.append(d)
        active_dots = sorted(filtered)

    return sorted(set(active_dots))


def classify_braille_image_no_ml(image: np.ndarray) -> str:
    """
    Classifica uma única imagem Braille usando apenas lógica de matriz.
    """
    # Mantido por compatibilidade, mas substituído no fluxo principal por método de protótipos estatísticos.
    dots = image_to_braille_dots(image)
    mat = dots_to_matrix(dots)
    for label, ref_mat in BRAILLE_MATRIX_MAP.items():
        if np.array_equal(mat, ref_mat):
            return label
    return "?"


def extract_cell_features(cell: np.ndarray) -> List[float]:
    """Extrai features estatísticas simples de uma célula Braille.
    FROM SCRATCH: intensidade média, desvio, proporções claro/escuro, entropia.
    """
    mean_int = float(cell.mean())
    std_int = float(cell.std())
    dark_ratio = float(np.sum(cell < 80) / cell.size)
    light_ratio = float(np.sum(cell > 170) / cell.size)
    # Histograma normalizado para entropia
    hist = cv2.calcHist([cell], [0], None, [32], [0,256]).flatten()
    hist /= (hist.sum() + 1e-8)
    entropy = float(-np.sum(hist * np.log(hist + 1e-8)))
    return [mean_int, std_int, dark_ratio, light_ratio, entropy]


def extract_image_features(image: np.ndarray) -> np.ndarray:
    """Extrai vetor de features para toda a imagem (6 células * 5 features)."""
    eq = cv2.equalizeHist(image)
    h, w = eq.shape
    cell_h = h // 3
    cell_w = w // 2
    feats: List[float] = []
    for row in range(3):
        for col in range(2):
            cell = eq[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w]
            feats.extend(extract_cell_features(cell))
    return np.array(feats, dtype=np.float32)


def process_single_image(img: np.ndarray) -> str:
    """Helper para paralelização."""
    return classify_braille_image_no_ml(img)


def visualize_samples(X_test: np.ndarray, y_test: np.ndarray, 
                     y_pred: np.ndarray, n_samples: int = 10):
    """
    Visualiza amostras do teste com predições.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Amostras do Dataset de Teste (Método sem ML)', fontsize=16)
    
    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            i = indices[idx]
            ax.imshow(X_test[i], cmap='gray')
            color = 'green' if y_pred[i] == y_test[i] else 'red'
            ax.set_title(f'True: {y_test[i]}\nPred: {y_pred[i]}', 
                        color=color, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    results_dir = Path('results/no_ml')
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / 'results_no_ml_samples.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualização salva em: {out_path}")
    plt.close()


def evaluate_no_ml():
    """
    Avalia o método sem ML no conjunto de teste e imprime métricas.
    Usa paralelização para acelerar processamento.
    """
    print("="*60)
    print("CLASSIFICAÇÃO BRAILLE - MÉTODO SEM APRENDIZADO DE MÁQUINA")
    print("="*60)
    print(f"\n1. Carregando dataset de: {DATASET_DIR.resolve()}")
    
    X_train, X_test, y_train, y_test = load_braille_dataset()
    
    print(f"   ✓ Total de imagens: {len(X_train) + len(X_test)}")
    print(f"   ✓ Treino: {len(X_train)} imagens")
    print(f"   ✓ Teste: {len(X_test)} imagens")
    print(f"   ✓ Classes únicas: {sorted(set(y_test))}")
    
    print("\n2. Construindo protótipos estatísticos (método sem ML)...")
    # Extrair features do conjunto de treino
    train_features = np.array([extract_image_features(img) for img in X_train])
    classes = sorted(set(y_train))
    prototypes: Dict[str, np.ndarray] = {}
    for cls in classes:
        cls_feats = train_features[y_train == cls]
        prototypes[cls] = cls_feats.mean(axis=0)
    print(f"   ✓ Protótipos gerados para {len(prototypes)} classes")

    print(f"\n3. Classificando {len(X_test)} imagens de teste por distância de protótipo...")
    test_features = np.array([extract_image_features(img) for img in X_test])
    y_pred_list: List[str] = []
    for idx, feats in enumerate(test_features):
        best_label = None
        best_dist = 1e12
        for cls, proto in prototypes.items():
            dist = np.linalg.norm(feats - proto)
            if dist < best_dist:
                best_dist = dist
                best_label = cls
        y_pred_list.append(best_label if best_label is not None else "?")
        if (idx+1) % 50 == 0 or (idx+1) == len(test_features):
            print(f"   Processado: {idx+1}/{len(test_features)}")
    y_pred = np.array(y_pred_list)

    print("\n" + "="*60)
    print("4. RESULTADOS")
    print("="*60)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n   ★ ACURÁCIA: {acc * 100:.2f}%")
    
    print("\n   Matriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n   Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Visualização
    print("\n5. Gerando visualizações...")
    visualize_samples(X_test, y_test, y_pred, n_samples=10)
    
    # Estatísticas adicionais
    correct = np.sum(y_test == y_pred)
    total = len(y_test)
    print(f"\n6. Estatísticas Finais:")
    print(f"   - Corretas: {correct}/{total}")
    print(f"   - Incorretas: {total - correct}/{total}")
    print(f"   - Taxa de erro: {(1 - acc) * 100:.2f}%")
    
    print("\n" + "="*60)
    print("PROCESSAMENTO CONCLUÍDO!")
    print("="*60)


if __name__ == "__main__":
    evaluate_no_ml()
