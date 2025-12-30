
# ✍️🎧 Multimodal Handwritten Mathematical Expression Recognition

This project implements a Transformer-based multimodal system for recognizing handwritten mathematical expressions using:

- ✏️ Stroke data (InkML)
- 🎧 Spoken mathematical expressions (Audio)

The model processes both modalities in parallel, fuses them, and generates LaTeX expressions using a Transformer decoder.

---

## ✨ Features

- Stroke-only handwritten math recognition  
- Multimodal learning (Stroke + Audio)  
- Parallel Transformer encoders  
- MFCC-based audio processing  
- End-to-end LaTeX generation  
- Token-level and expression-level evaluation  

---

## 🧠 Model Architecture

```
InkML → Stroke Encoder ┐
                       ├─► Fusion → Transformer Decoder → LaTeX
Audio → MFCC → Encoder ┘
```

---

## ▶️ Run the Model

### Stroke Only
```bash
python src/stroke_only/main_stroke.py
```

### Stroke + Audio
```bash
python src/stroke_audio/main_multimodal.py
```

---

## 📊 Results

| Model | Token Accuracy | Expression Accuracy |
|------|----------------|---------------------|
| Stroke Only | 35.47% | 7.32% |
| Stroke + Audio | 74.20% | 24.80% |

---

## 🧪 Sample Predictions

| Sample | Ground Truth | Prediction |
|------|--------------|------------|
| 1 | y₀ ≈ √5 | y₀ ≈ √5 |
| 2 | (2+20)^{√9/√8} | (2+22)^{√9·√4} |
| 3 | ṙ = −G/2 · u̇ | ṙ = −vec(h)/u |
| 4 | 1/ρ² ∇ρ × ∇p | 1/a (√a)⁻¹ |
| 5 | μ = 2a√(2/π) | μ = 2√(2/π) |

---

## 🚀 Future Work

- Attention visualization
- Whisper / HuBERT audio encoders
- Beam search decoding
- Web-based demo
- Real-time handwriting recognition
