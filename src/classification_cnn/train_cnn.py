# src/classification_cnn/train_cnn.py

from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import sys

# Garante import absoluto de 'src' quando executado via caminho relativo.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.pipeline import preprocess_for_braille


IMG_SIZE = (64, 64)
DATASET_DIR = Path("dataset")
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PATH = Path("models/braille_cnn.h5")


def configure_gpu():
    """Detecta GPUs, configura memory growth e imprime detalhes úteis."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("[GPU] Nenhuma GPU detectada. Usando CPU.")
            return
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        logical = tf.config.list_logical_devices("GPU")
        print(f"[GPU] Físicas: {len(gpus)} | Lógicas: {len(logical)}")
        for idx, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            name = details.get('device_name', 'Unknown')
            cc = details.get('compute_capability', 'n/a')
            print(f"[GPU] {idx}: name={name} compute_capability={cc}")
    except Exception as e:
        print(f"[GPU] Erro ao configurar: {e}")


def infer_label_from_path(path: Path) -> str:
    """Extrai a primeira letra alfabética do nome do arquivo.
    Evita assumir estrutura de pastas (dataset plano)."""
    for ch in path.name:
        if ch.isalpha():
            return ch.upper()
    return "?"


def load_braille_dataset(
    dataset_dir: Path = DATASET_DIR,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    image_paths: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(dataset_dir.rglob(ext))

    X = []
    y = []

    for path in image_paths:
        img_raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            continue
        # Aplica pipeline padronizada (A). Mantém saída grayscale equalizada.
        img_proc = preprocess_for_braille(img_raw, resize=img_size, return_binary=False)
        X.append(img_proc)
        y.append(infer_label_from_path(path))

    # Debug distribuição de labels
    counts = Counter(y)
    print("[DEBUG] Distribuição de labels carregadas (CNN):")
    for lbl in sorted(counts.keys()):
        print(f"  - {lbl}: {counts[lbl]}")
    print(f"[DEBUG] Total distintos: {len(counts)}")

    X = np.array(X, dtype=np.float32) / 255.0
    X = np.expand_dims(X, axis=-1)  # (N, H, W, 1)

    y = np.array(y)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cat,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    return X_train, X_test, y_train, y_test, le


def build_cnn(input_shape: Tuple[int, int, int], num_classes: int, variant: str = "lenet") -> Sequential:
    """Constroi CNN (variant 'lenet' ou 'deep'). Usa data augmentation embutida.
    A versão 'lenet' é menor para reduzir overfitting no dataset pequeno.
    """
    aug = Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.15),
    ], name="augmentation")

    model = Sequential(name=f"cnn_{variant}")
    model.add(layers.Input(shape=input_shape))
    model.add(aug)

    if variant == "lenet":
        # LeNet-like simplificado
        model.add(layers.Conv2D(32, (5,5), activation="relu", padding="same"))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Conv2D(64, (5,5), activation="relu", padding="same"))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Conv2D(128, (3,3), activation="relu", padding="same"))
        model.add(layers.MaxPool2D((2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.5))
    else:
        # Profunda (original simplificada para reduzir overfitting)
        for filters in (64,128,256):
            model.add(layers.Conv2D(filters, (3,3), activation="relu", padding="same"))
            model.add(layers.BatchNormalization())
            model.add(layers.Conv2D(filters, (3,3), activation="relu", padding="same"))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPool2D((2,2)))
            model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(variant: str = "lenet", use_pipeline: bool = True):
    configure_gpu()
    print("Carregando dataset...")
    X_train, X_test, y_train, y_test, le = load_braille_dataset()

    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    print(f"Input shape: {input_shape}, num_classes: {num_classes}")

    print(f"Construindo CNN variante={variant}...")
    model = build_cnn(input_shape, num_classes, variant=variant)
    model.summary()

    print("Treinando modelo...")
    
    # Callbacks para otimizar treinamento
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("Avaliando no conjunto de teste...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Acurácia (CNN): {test_acc * 100:.2f}%")

    # Predições para relatório
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test_labels, y_pred_labels)
    print("\nMatriz de confusão:")
    print(cm)

    report_str = classification_report(
        y_test_labels,
        y_pred_labels,
        target_names=le.classes_,
    )
    print("\nRelatório de classificação:")
    print(report_str)

    # Salvar artefatos em results/cnn
    results_dir = Path("results/cnn")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Matriz de confusão como imagem
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Matriz de Confusão - CNN")
        ax.set_xticks(range(len(le.classes_)))
        ax.set_yticks(range(len(le.classes_)))
        ax.set_xticklabels(le.classes_, rotation=90)
        ax.set_yticklabels(le.classes_)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                if val:
                    ax.text(j, i, val, ha='center', va='center', color='black', fontsize=6)
        plt.tight_layout()
        fig.savefig(results_dir / "confusion_matrix_cnn.png", dpi=150)
        plt.close(fig)
        print(f"[cnn] Matriz de confusão salva em {results_dir / 'confusion_matrix_cnn.png'}")
    except Exception as e:
        print(f"[cnn] Falha ao salvar matriz de confusão: {e}")

    # Salvar relatório e acurácia
    # Curvas de treinamento (loss / accuracy)
    try:
        import matplotlib.pyplot as plt
        fig2, ax2 = plt.subplots(1,2, figsize=(10,4))
        ax2[0].plot(history.history['loss'], label='train')
        ax2[0].plot(history.history['val_loss'], label='val')
        ax2[0].set_title('Loss')
        ax2[0].legend()
        ax2[1].plot(history.history['accuracy'], label='train')
        ax2[1].plot(history.history['val_accuracy'], label='val')
        ax2[1].set_title('Accuracy')
        ax2[1].legend()
        fig2.tight_layout()
        fig2.savefig(results_dir / "training_curves.png", dpi=150)
        plt.close(fig2)
        print(f"[cnn] Curvas salvas em {results_dir / 'training_curves.png'}")
    except Exception as e:
        print(f"[cnn] Falha ao salvar curvas: {e}")

    with open(results_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Variante: {variant}\n")
        f.write(f"Acurácia teste: {test_acc*100:.2f}%\n")
        f.write(report_str + "\n")
    print(f"[cnn] Métricas salvas em {results_dir / 'metrics.txt'}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH.resolve()}")

    # Salvar LabelEncoder para uso na webcam
    le_path = MODEL_PATH.parent / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"LabelEncoder salvo em: {le_path.resolve()}")


if __name__ == "__main__":
    main()