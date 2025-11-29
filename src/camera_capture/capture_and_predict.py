# src/camera_capture/capture_and_predict.py

from pathlib import Path
import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
import sys
from pathlib import Path as _PathForSys
ROOT_DIR = _PathForSys(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.classification_no_ml.infer_no_ml import classify_frame as classify_no_ml
from src.preprocessing.pipeline import preprocess_for_braille

IMG_SIZE = (64, 64)
MODEL_PATH = Path("models/braille_cnn.h5")
LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")  # salvaremos depois


def load_label_encoder() -> LabelEncoder:
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return le


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Recebe frame BGR da webcam e devolve tensor (1, H, W, 1) normalizado.
    Aqui você pode, se quiser, recortar ROI manualmente depois.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)
    gray = gray.astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    return gray


def parse_args():
    p = argparse.ArgumentParser(description="Webcam Braille Capture")
    p.add_argument("--method", choices=["cnn", "no-ml"], default="cnn", help="Método de classificação")
    p.add_argument("--roi", action="store_true", help="Aplicar detecção simples de ROI")
    return p.parse_args()


def main():
    args = parse_args()
    method = args.method
    use_roi = args.roi
    print(f"[webcam] Método: {method} ROI={use_roi}")

    if method == "cnn":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
        if not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(f"LabelEncoder não encontrado em {LABEL_ENCODER_PATH}")
        print("Carregando modelo e LabelEncoder (CNN)...")
        model = load_model(MODEL_PATH)
        le = load_label_encoder()
    else:
        model = None
        le = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")

    print("Pressione 'c' para capturar e classificar. 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam - Braille", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            if method == "cnn":
                input_tensor = preprocess_frame(frame if not use_roi else preprocess_for_braille(frame, resize=IMG_SIZE, return_binary=False, use_roi=True))
                preds = model.predict(input_tensor)
                class_idx = np.argmax(preds, axis=1)[0]
                label = le.inverse_transform([class_idx])[0]
                score = float(np.max(preds))
                dots_info = "CNN"
            else:
                pred, conf, dots = classify_no_ml(frame if not use_roi else preprocess_for_braille(frame, resize=(128,128), return_binary=False, use_roi=True))
                label = pred
                score = conf
                dots_info = f"dots={dots}" if dots else "dots=[]"

            print(f"Predição: {label} | Score: {score:.4f} | {dots_info}")
            out = frame.copy()
            cv2.putText(
                out,
                f"{label} ({score:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Resultado", out)
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
