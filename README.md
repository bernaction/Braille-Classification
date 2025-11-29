# 📘 Braille Classification (Opção 1)

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green.svg)

**Disciplina:** Processamento de Imagens — UNIVALI  
**Professor:** Felipe Viel  
**Alunos:** *Bernardo Vannier* e *André Goedert*  


---

# 🟦 Projeto: Classificação de Caracteres Braille usando Processamento de Imagens e Aprendizado de Máquina

Este repositório contém o desenvolvimento completo do **Projeto Final (Opção 1)** da disciplina **Processamento de Imagens**, cujo objetivo é comparar duas abordagens diferentes para reconhecimento de caracteres Braille:

`Aprendizado de Máquina, uma subárea da inteligência artificial, tem se destacado por sua
capacidade de aprender padrões e realizar tarefas complexas a partir de dados. Uma aplicação
promissora dessa tecnologia é o reconhecimento de Braille, um sistema de leitura e escrita utilizado por
pessoas com deficiência visual. O Braille, inventado por Louis Braille no século XIX, consiste em um
conjunto de pontos em relevo que representam letras, números e símbolos. O desafio de transformar
essas elevações em texto legível por máquinas é significativo, exigindo algoritmos sofisticados e dados
extensivos para treinar modelos capazes de interpretar com precisão as variações táteis.

Nessa tarefa, você precisará fazer o seguinte (para o item 1 ao 4 a nota máxima será 9,0. O ponto final
será dado caso o item 5 for feito e for “funcional”):
1. Aplicar os filtros e pré-processamentos necessários ou que vocês achem relevantes nas imagens
de entrada. Aqui podem ser usados algoritmos da OpenCV ou biblioteca equivalente.
2. Comparar duas técnicas: sem aprendizado de máquina (1) ou com aprendizado de máquina (2).
Sem aprendizado de máquina, pode ser aplicado histograma acumulativo e lógica de matriz.
Com aprendizado de máquina, aplicar CNN ou modelo de rede neural da sua preferência. Em
(1), deverá ser apresentado os algoritmos de histograma acumulativo e lógica de matriz from
Scratch. Para CNN (ou equivalente), poderá ser utilizado modelo prontos ou pré-treinados como
LeNet, VGG, YOLO e afins.
3. Para o trabalho, usar o dataset disponível no github da disciplina. Caso o dataset não esteja
separado em teste e treino, assuma 20% do dataset para teste e 80% para treino.
4. Vocês devem testar e mostrar a acurácia para teste.
5. Somente para a opção (1) e (2): Após o algoritmo de reconhecimento estar pronto, deve ser
implementado uma função que abra a câmera (por exemplo, a notebook) tire uma foto e faça o
reconhecimento.`

1. **Método sem aprendizado de máquina:**  
   - Histograma acumulativo  
   - Lógica de matriz (from scratch)

2. **Método com aprendizado de máquina:**  
   - Rede neural convolucional (CNN)  
   - Utilizando TensorFlow/Keras  
   - Possibilidade de usar modelos simples como LeNet ou CNN customizada  

Ambas as abordagens são testadas e comparadas usando o dataset fornecido pelo professor.

---

# 🎯 Resultados Alcançados

| Método | Acurácia Teste | Observações |
|--------|----------------|-------------|
| **CNN (LeNet + Aug)** | **95.51%** ✅ | Superou meta de 75% |
| **Sem ML (Hist + Matriz)** | 37.82% | Baseline pedagógico |

📊 **Relatório Completo:** [RESULT.md](RESULT.md)

### Destaques
- ✅ Pipeline de preprocessamento padronizado
- ✅ Data augmentation (rotação, zoom, contraste)
- ✅ Inferência via webcam para ambos os métodos
- ✅ Matrizes de confusão e curvas de treinamento
- ✅ Implementação from scratch do histograma acumulativo

---

# 📂 Dataset

- Fonte: GitHub da disciplina  
  https://github.com/VielF/ColabProjects/tree/main/dataset/DatasetBraile/Option1  
- O conjunto contém imagens de caracteres Braille (A–Z).  
- Como o dataset não está separado:
  - **80%** para treino  
  - **20%** para teste  

As imagens também são usadas para inferência via webcam.

---

# 🧩 Estrutura do Projeto

```text
├── dataset/                          # Dataset Braille (A-Z, 60 imagens/letra)
├── src/
│   ├── preprocessing/
│   │   ├── pipeline.py              # Pipeline padronizado (Gaussian, Equalização, Morfologia)
│   │   └── histogram_utils.py       # Histograma acumulativo from scratch
│   ├── classification_no_ml/
│   │   ├── simple_braille.py        # Classificador baseline (treino/teste)
│   │   └── infer_no_ml.py           # Inferência sem ML para webcam
│   ├── classification_cnn/
│   │   └── train_cnn.py             # CNN LeNet (variante otimizada)
│   └── camera_capture/
│       └── capture_and_predict.py   # Webcam com seleção de método
├── models/
│   ├── braille_cnn.h5               # Modelo treinado (95.51% acc)
│   └── label_encoder.pkl            # Encoder de labels
├── results/
│   ├── cnn/                         # Confusion matrix, curvas, métricas
│   ├── no_ml/simple/                # Resultados baseline
│   ├── histograms/                  # Análises de histograma
│   └── preprocessing/               # Demos preprocessamento
├── RESULT.md                        # Relatório completo A-E
├── requirements.txt
└── README.md
```

---

# ⚙️ Etapas do Desenvolvimento

## ✔️ 1. Pré-processamento das Imagens

Pipeline padronizado implementado (`src/preprocessing/pipeline.py`):

```python
1. Conversão para grayscale
2. GaussianBlur (5×5) → suavização preservando bordas
3. Equalização de histograma → melhora contraste
4. Morfologia (fechamento, kernel elipse 3×3) → realça pontos
5. Binarização Otsu (para método sem ML) ou normalização [0,1] (para CNN)
6. (Opcional) Detecção de ROI via maior contorno
```

**Saídas:**
- Método sem ML: imagem binária (dots brancos sobre fundo preto)
- CNN: imagem grayscale normalizada + data augmentation

---

## ✔️ 2. Método sem Aprendizado de Máquina (Histograma + Lógica de Matriz)

**Implementação from scratch** (`src/classification_no_ml/simple_braille.py`):

### Algoritmo
```python
1. Dividir imagem binária em grade 3×2 (padrão Braille)
2. Para cada célula:
   a. Calcular histograma manual: hist[intensidade] += 1
   b. Histograma acumulativo: cum_hist = np.cumsum(hist)
   c. Score = 1 - (cum_hist[200] / total_pixels)  # proporção pixels ≥200
3. Threshold adaptativo: thr = max(0.15, mean_score × 0.85)
4. Ativar dots com score ≥ thr → gerar matriz 3×2 binária
5. Comparar com dicionário Braille {A:[1], B:[1,2], ..., Z:[1,3,5,6]}
```

**Acurácia:** 37.82% (esperado para baseline heurístico)  

---

## ✔️ 3. Método com Aprendizado de Máquina (CNN)

**Arquitetura LeNet Otimizada** (`src/classification_cnn/train_cnn.py`):

```python
Augmentation (RandomRotation, Zoom, Contrast)
  ↓
Conv2D(32, 5×5) + ReLU + MaxPool(2×2)
  ↓
Conv2D(64, 5×5) + ReLU + MaxPool(2×2)
  ↓
Conv2D(128, 3×3) + ReLU + MaxPool(2×2)
  ↓
Flatten → Dense(256, ReLU) + Dropout(0.5)
  ↓
Softmax(26 classes)
```

**Treinamento:**
- Optimizer: Adam (lr=1e-3)
- Callbacks: EarlyStopping, ReduceLROnPlateau
- Split: 80% treino, 20% teste (estratificado)
- Epochs: 50 (early stop ~epoch 50)

**Acurácia:** 95.51% ✅ (supera 75% exigido)  

---

## ✔️ 4. Acurácia e Resultados

### Métricas Geradas

**CNN:**
- Acurácia: **95.51%** (312 amostras teste)
- Precision/Recall/F1: ~0.96 (macro avg)
- Artefatos: `results/cnn/`
  - `confusion_matrix_cnn.png` – matriz 26×26
  - `training_curves.png` – loss/accuracy por época
  - `metrics.txt` – classification report completo

**Sem ML:**
- Acurácia: **37.82%**
- Artefatos: `results/no_ml/simple/`
  - `confusion_matrix_simple.png`
  - `summary.txt`

**Análise Comparativa:** Ver [RESULT.md](RESULT.md) para discussão detalhada (+57pp de melhoria CNN vs baseline).

---

## ✔️ 5. Inferência via Webcam

**Funcionalidades** (`src/camera_capture/capture_and_predict.py`):
- Seleção de método: `--method cnn` ou `--method no-ml`
- Opção ROI automático: `--roi`
- Tecla `c` → captura e classifica
- Tecla `q` → sair
- Exibe: caractere reconhecido + score de confiança

**Comandos:**
```bash
# CNN (95% acurácia)
python src/camera_capture/capture_and_predict.py --method cnn

# Sem ML (baseline)
python src/camera_capture/capture_and_predict.py --method no-ml --roi
```

---


# ▶️ Como Executar o Projeto

## 1️⃣ Setup Inicial

```bash
# Clonar repositório
git clone <repo-url>
cd Braille-Classification

# Criar ambiente virtual Python 3.11+
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## 2️⃣ Baixar Dataset

```bash
# Baixar de: https://github.com/VielF/ColabProjects/tree/main/dataset/DatasetBraile/Option1
# Extrair para pasta dataset/ (60 imagens × 26 letras = 1560 total)
```

## 3️⃣ Treinar Modelos

```bash
# Método sem ML (baseline)
python src/classification_no_ml/simple_braille.py
# → Gera: results/no_ml/simple/

# CNN (LeNet otimizada)
python src/classification_cnn/train_cnn.py
# → Gera: models/braille_cnn.h5, results/cnn/
```

## 4️⃣ Inferência Webcam

```bash
# Com CNN (recomendado)
python src/camera_capture/capture_and_predict.py --method cnn

# Com método sem ML
python src/camera_capture/capture_and_predict.py --method no-ml
```

## 5️⃣ Análise de Histogramas (Opcional)

```bash
python src/preprocessing/histogram_utils.py
# → Gera: results/histograms/
```

---

# 🛠️ Tecnologias Utilizadas

| Categoria | Ferramentas |
|-----------|-------------|
| **Linguagem** | Python 3.11 |
| **Deep Learning** | TensorFlow 2.16, Keras |
| **Visão Computacional** | OpenCV |
| **Computação Científica** | NumPy, SciPy |
| **ML & Métricas** | scikit-learn |
| **Visualização** | Matplotlib |
| **Ambiente** | pyenv, venv |
| **Hardware** | NVIDIA TITAN V (opcional, CUDA 11.8) |

---

# 📚 Referências

- [OpenCV Documentation](https://docs.opencv.org/)
- [Keras API Reference](https://keras.io/api/)
- [Dataset Original](https://github.com/VielF/ColabProjects/tree/main/dataset/DatasetBraile/Option1)
- LeNet-5: LeCun et al., "Gradient-Based Learning Applied to Document Recognition"

---

# 📄 Licença

Projeto acadêmico – UNIVALI 2025