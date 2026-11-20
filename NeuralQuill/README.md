# Shakespeare Generator
### Character-level Transformer — Built from Scratch

> NumPy only · No PyTorch · No GPU · No API

---

## What it does
Trains a Transformer neural network on Shakespeare text.
After training, it generates new Shakespeare-style writing
character by character.

## Project structure
```
shakespeare/
├── input.txt          ← Shakespeare training corpus
├── model.py           ← Transformer architecture (from scratch)
├── train.py           ← Training script
├── generate.py        ← CLI text generator
├── app.py             ← Web UI (Flask)
├── requirements.txt   ← numpy, flask only
└── README.md
```
(tokenizer.pkl and model.pkl are created after training)

## Quick start

### 1. Install dependencies
```bash
pip install numpy flask
```

### 2. Train the model
```bash
python3 train.py
# ~5 minutes on CPU for 5000 iterations
# loss drops from ~4.2 → ~1.4
```

### 3. Generate text (CLI)
```bash
python3 generate.py --prompt "HAMLET:"
python3 generate.py --prompt "To be, or" --length 500 --temp 0.9
python3 generate.py --temp 0.6 --top_k 20
```

### 4. Launch web UI
```bash
python3 app.py
# open http://localhost:5000
```

---

## Architecture
```
Input characters
      ↓
Character Tokenizer  (60 unique chars → integers)
      ↓
Token Embedding      (V × 128)
+
Position Embedding   (128 × 128)
      ↓
4 × Transformer Block
  ├─ LayerNorm
  ├─ Multi-Head Causal Self-Attention (4 heads)
  │    Scaled dot-product attention
  │    Causal mask (no future peeking)
  ├─ Residual connection
  ├─ LayerNorm
  ├─ Feed-Forward (128 → 512 → 128, ReLU)
  └─ Residual connection
      ↓
Final LayerNorm
      ↓
LM Head (128 → vocab_size)
      ↓
Softmax → sample next character
```

## Training details
```
Parameters   : ~800K
Optimizer    : Adam (β1=0.9, β2=0.999)
LR schedule  : warmup + cosine decay
Grad clipping: 1.0
Block size   : 128 chars context
Iterations   : 5000 (default)
Training time: ~5 min on CPU
```

## Generation settings
| Setting     | Effect |
|-------------|--------|
| Temperature | Low (0.5) = focused/repetitive · High (1.2) = creative/chaotic |
| Top-K       | Low (10) = safe choices · High (80) = more variety |
| Length      | 100–1000 characters |

## Tips for better output
- Use more training data (repeat input.txt content)
- Train longer: `python3 train.py --iters 10000`
- Bigger model: `python3 train.py --n_embd 256 --n_layer 6`
- Low temperature (0.6) for coherent text
- Start prompt with a character name: `HAMLET:\n`
