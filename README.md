# Face Expression Recognition With Fuzzy Logic 🎭

<div align="center">

**Sistema de Reconhecimento de Expressões Faciais usando YOLOv11 e Lógica Fuzzy**

[![Python](https://img.shields.io/badge/Python-3.13.5+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)](STATUS.md)
</div>

---

## 📖 Sobre o Projeto

Este projeto implementa um sistema completo de **reconhecimento de expressões faciais** combinando técnicas de Deep Learning e Lógica Fuzzy. O sistema detecta 68 landmarks faciais usando uma arquitetura YOLOv11 customizada e classifica emoções através de um sistema fuzzy baseado em deformações geométricas.

Este projeto é parte de um Trabalho de Conclusão de Curso (TCC) da **Universidade Estadual do Oeste do Paraná (Unioeste)**.


### 🎯 Objetivos

- ✅ Detectar com precisão 68 landmarks faciais (padrão 300W/iBUG)
- ✅ Calcular deformações geométricas em relação a uma face neutra
- ✅ Classificar expressões faciais usando lógica fuzzy
- ✅ Fornecer uma solução interpretável e robusta

### 🏆 Destaques

- **Arquitetura YOLOv11:** Estado da arte em detecção de objetos adaptada para landmarks
- **Lógica Fuzzy:** Sistema interpretável com regras linguísticas
- **Wing Loss:** Loss especializada para landmarks faciais
- **Multi-escala:** Detecção em múltiplas resoluções (P3, P4, P5)
- **Atenção Espacial:** Módulos C2PSA para melhor precisão

---

## 🏗️ Arquitetura do Sistema

```
┌──────────────────────┐
│   INPUT IMAGE        │
│   (3 x 640 x 640)   │
└──────────┬───────────┘
           │
    ┌──────▼──────┐
    │  YOLOv11    │
    │  Backbone   │
    └──────┬──────┘
           │
  ┌────────┼────────┐
  │        │        │
┌─▼─┐    ┌─▼─┐    ┌─▼─┐
│P3 │    │P4 │    │P5 │
└─┬─┘    └─┬─┘    └─┬─┘
  │        │        │
  └────────┼────────┘
           │
    ┌──────▼──────┐
    │  YOLOv11    │
    │  Neck       │
    │  (PANet)    │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Detection   │
    │   Head      │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │68 Landmarks │
    │(x, y, conf) │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │Deformações  │
    │Geométricas  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │Sistema Fuzzy│
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   EMOÇÃO    │
    └─────────────┘
```

---

## 📊 Estrutura dos Landmarks

68 pontos do padrão 300W/iBUG:

| Região               | Índices    | Descrição                                      |
|:--------------------|:-----------|:-----------------------------------------------|
| Contorno facial      | **1–17**  | Mandíbula e formato geral do rosto             |
| Sobrancelha esquerda | **18–22** | Arco e posição relativa                        |
| Sobrancelha direita  | **23–27** | Arco e posição relativa                        |
| Nariz                | **28–36** | 28–31: ponte / 32–36: base e narinas           |
| Olho esquerdo        | **37–42** | Pálpebra, abertura e direção do olhar          |
| Olho direito         | **43–48** | Pálpebra, abertura e direção do olhar          |
| Boca                 | **49–68** | Lábios superior e inferior, abertura e sorriso |

---

## 🚀 Pipeline de Processamento

### 1️⃣ YOLOv11 → Detecção de Landmarks

A rede YOLOv11 customizada detecta diretamente os 68 landmarks faciais:

- **Entrada:** Imagem RGB (640×640)
- **Saída:** Matriz de coordenadas (68 × 3)
  - `x, y`: Coordenadas normalizadas [0, 1]
  - `confidence`: Score de confiança da detecção

**Componentes:**
- **Backbone:** Extração de features com blocos C3K2 e SPFF
- **Neck:** PANet com atenção espacial (C2PSA)
- **Head:** Detecção multi-escala com fusão de predições

### 2️⃣ Cálculo de Deformações Geométricas

A partir dos landmarks, calcula-se:
- **Distâncias euclidianas** entre pontos chave
- **Ângulos relativos** (sobrancelhas, boca)
- **Relações normalizadas** comparando com face neutra

**Fórmula:**

$$d_{rel}(i, j) = \frac{ \| p_i - p_j \| - \| p_i^0 - p_j^0 \| }{ \| p_i^0 - p_j^0 \| }$$

Onde:
- $p_i = (x_i, y_i)$ → coordenadas do ponto $i$ na **imagem atual**
- $p_i^0 = (x_i^0, y_i^0)$ → coordenadas do ponto $i$ na **face neutra**
- $\| p_i - p_j \| = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$ → distância euclidiana

**Interpretação:**
- $d_{rel} > 0$ → **Expansão** (aumento da distância)
- $d_{rel} < 0$ → **Contração** (redução da distância)
- $d_{rel} \approx 0$ → **Sem variação** significativa

### 3️⃣ Mapeamento Fuzzy das Deformações

Variáveis linguísticas fuzzy:
- `abertura_boca` → {baixa, média, alta}
- `elevacao_sobrancelha` → {baixa, média, alta}
- `abertura_olhos` → {baixa, média, alta}

**Regras Fuzzy (exemplos):**
```
SE (boca_aberta = alta) E (olhos_abertos = altos) ENTÃO expressão = SURPRESA
SE (boca_contraída = alta) E (sobrancelha_abaixada = alta) ENTÃO expressão = RAIVA
SE (boca_sorriso = alto) E (olhos_semicerrados = médios) ENTÃO expressão = FELICIDADE
```

### 4️⃣ Inferência e Decisão

O sistema fuzzy gera scores (0–1) para cada emoção usando:
- **Método de Inferência:** Mamdani ou Sugeno
- **Defuzzificação:** Centro de gravidade
- **Saída:** Classe emocional + confidence

---

## 🛠️ Instalação

### Pré-requisitos

- Python 3.13.5+
- CUDA 11.8+ (opcional, para GPU)

### Instalação Rápida

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/Fer-With-Fuzzy.git
cd Fer-With-Fuzzy

# Criar ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar dependências
pip install -r requirements.txt
```

---

## 💻 Exemplos de Uso

### Treinamento

```bash
# Básico
python main.py --mode train --data-dir datasets/300W --epochs 100

# Com configurações customizadas
python main.py \
    --mode train \
    --data-dir datasets/300W \
    --epochs 200 \
    --batch-size 32 \
    --lr 0.001 \
    --base-channels 64
```

### Teste/Avaliação

```bash
python main.py \
    --mode test \
    --weights checkpoints/best_model.pth \
    --data-dir datasets/300W
```

### Demo/Inferência

```bash
python main.py \
    --mode demo \
    --image examples/face.jpg \
    --weights checkpoints/best_model.pth \
    --output results/face_landmarks.jpg
```

### Uso Programático

```python
import torch
from core import YOLOv11LandmarkDetector

# Carregar modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv11LandmarkDetector().to(device)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predição
with torch.no_grad():
    landmarks, confidence, mask = model.predict_landmarks(image_tensor)

print(f"Landmarks detectados: {mask.sum()}/68")
```

---

## 📂 Estrutura do Projeto

```
Fer-With-Fuzzy/
├── blocks/              # Blocos da rede (C3K2, C2PSA, SPFF)
├── core/               # Arquitetura principal (Backbone, Neck, Head)
├── utils/              # Dataset loader, Loss functions
├── config/             # Configurações do sistema
├── tests/              # Testes unitários
├── datasets/           # Datasets (300W)
├── checkpoints/        # Modelos treinados
├── main.py            # Script principal
├── README.md          # Este arquivo
├── USAGE.md           # Guia detalhado
└── STATUS.md          # Status do projeto
```

---

## 👥 Autores

- **Desenvolvedor:** Felipe Kravec Zanatta
- **Orientador:** Adriana Postal
- **Instituição:** Unioeste - Universidade Estadual do Oeste do Paraná
- **Curso:** Ciência da Computação
- **Ano:** 2025
---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela!**


</div>
