# How to Run Sparse Autoencoder Analysis

## Overview
This system trains a Sparse Autoencoder to interpret your PPO trading agent's decisions by learning which features it pays attention to.

---

## 📋 Prerequisites

1. **PPO Training Data** ✅ (already collected):
   - File: `data/custom.npz`
   - Contains: 230 samples with observations (39,964-dim) and policy outputs from your PPO agent
   - You already have this file!

2. **Python Dependencies**:
   ```bash
   pip install torch numpy scikit-learn
   ```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Train the Sparse Autoencoder
```bash
python train_ae.py --data-path data/custom.npz --output-dir checkpoints/large_run --model-type topk --latent-dim 128 --k 32 --epochs 100 --normalize-decoder --use-auxiliary --save-best
```

**Key arguments:**
- `--data-path`: Your PPO training data (observations + actions)
- `--output-dir`: Where to save the trained model
- `--latent-dim 128`: Number of latent features (increase for more capacity)
- `--k 32`: Sparsity level (32/128 = 25% active)
- `--epochs 100`: Training iterations

**What it does:**
- Trains a 128-latent TopK Sparse Autoencoder on PPO observations
- Takes ~10-15 minutes (100 epochs)
- Saves best model to: `checkpoints/large_run/sparse_ae_best.pt`
- Outputs training metrics every 10 epochs

**Expected output:**
```
Epoch 10: train_r2=0.78, val_r2=0.75, sparsity=25%
Epoch 20: train_r2=0.82, val_r2=0.79, sparsity=25%
...
Epoch 100: train_r2=0.94, val_r2=0.83, sparsity=25%
✓ Training complete! Best model saved.
```

---

### Step 2: Compute Alignment to PPO Policy
```bash
python align_factors.py --data-path data/custom.npz --checkpoint checkpoints/large_run/sparse_ae_best.pt --output-dir analysis/alignment_large
```

**Key arguments:**
- `--data-path`: Same PPO data used for training
- `--checkpoint`: Path to your trained model
- `--output-dir`: Where to save alignment results

**What it does:**
- Measures how well learned latents predict PPO's actions
- Calculates R² score (target: >0.5 = excellent)
- Saves results to: `analysis/alignment_large/`

**Expected output:**
```
Computing alignment...
R² = 0.528 (53% of PPO decisions explained)
Active latents: 109/128
✓ Alignment: EXCELLENT
```

**Interpreting R² scores:**
- R² > 0.5: Excellent (production ready)
- R² 0.3-0.5: Good (usable)
- R² < 0.3: Poor (need more capacity or data)

---

### Step 3: Extract Interpretable Features
```bash
python extract_features.py --checkpoint checkpoints/large_run/sparse_ae_best.pt --data-path data/custom.npz --output-dir analysis/feature_extraction_large --ticker-file ../../tickers.csv
```

**Key arguments:**
- `--checkpoint`: Path to trained model
- `--data-path`: PPO data (for reference shapes)
- `--output-dir`: Where to save feature extraction results
- `--ticker-file`: CSV with stock ticker names

**What it does:**
- Maps each latent to its top input features (stock names, lags, prices)
- Generates human-readable feature names
- Saves to: `analysis/feature_extraction_large/`

**Expected output:**
```
Extracting features for 128 latents...

Latent 115 (most important):
  1. VOLTAS.NS_t-10_high (weight: 0.153)
  2. VOLTAS.NS_t-10_open (weight: 0.152)
  3. VOLTAS.NS_t-10_volume (weight: 0.149)
  → Interpretation: VOLTAS 10-day momentum signal

Latent 94:
  1. VOLTAS.NS_t-18_open (weight: 0.122)
  2. VOLTAS.NS_t-18_close (weight: 0.120)
  → Interpretation: VOLTAS swing trading signal
...

✓ Feature extraction complete!
```

---

## 📊 Understanding the Results

### Key Files Generated:
1. **`checkpoints/large_run/sparse_ae_best.pt`**
   - The trained model weights
   - Use this for inference on new data

2. **`analysis/alignment_large/alignment_metrics.json`**
   ```json
   {
     "r2_mean": 0.5284,
     "mse_mean": 0.088715,
     "active_latents": 109
   }
   ```

3. **`analysis/feature_extraction_large/top_features.json`**
   - Maps each latent to its top features
   - Human-readable feature names
   - Feature importance weights

---

## 🔧 Advanced Options

### Training with Different Hyperparameters

**Increase capacity for better alignment:**
```bash
python train_ae.py --data-path data/custom.npz --output-dir checkpoints/xlarge_run --model-type topk --latent-dim 256 --k 64 --epochs 150 --normalize-decoder --use-auxiliary --save-best
```

**Faster training (fewer epochs):**
```bash
python train_ae.py --data-path data/custom.npz --output-dir checkpoints/quick_run --model-type topk --latent-dim 128 --k 32 --epochs 50 --normalize-decoder --use-auxiliary --save-best
```

**Different sparsity levels:**
```bash
# More sparse (16/128 = 12.5% active)
python train_ae.py --data-path data/custom.npz --output-dir checkpoints/sparse_run --model-type topk --latent-dim 128 --k 16 --epochs 100 --normalize-decoder --use-auxiliary --save-best

# Less sparse (48/128 = 37.5% active)
python train_ae.py --data-path data/custom.npz --output-dir checkpoints/dense_run --model-type topk --latent-dim 128 --k 48 --epochs 100 --normalize-decoder --use-auxiliary --save-best
```

After training with custom settings, update the alignment and extraction commands:
```bash
# Alignment
python align_factors.py --data-path data/custom.npz --checkpoint checkpoints/YOUR_RUN/sparse_ae_best.pt --output-dir analysis/alignment_YOUR_RUN

# Feature extraction
python extract_features.py --checkpoint checkpoints/YOUR_RUN/sparse_ae_best.pt --data-path data/custom.npz --output-dir analysis/feature_extraction_YOUR_RUN --ticker-file ../../tickers.csv
```

### Using Different SAE Variants

**L1 regularization (alternative to TopK):**
```bash
python train_ae.py --data-path data/custom.npz --output-dir checkpoints/l1_run --model-type l1 --latent-dim 128 --sparsity-weight 0.01 --epochs 100 --normalize-decoder --save-best
```

**JumpReLU activation (gated sparsity):**
```bash
python train_ae.py --data-path data/custom.npz --output-dir checkpoints/jumprelu_run --model-type jumprelu --latent-dim 128 --epochs 100 --normalize-decoder --save-best
```

**Model type comparison:**
- **topk**: Best for interpretability, fixed sparsity (recommended) ✅
- **l1**: L1 penalty, variable sparsity per sample
- **jumprelu**: Learned gating, dynamic sparsity

---

## 📈 What You Learn

After running all 3 steps, you'll know:

1. **Which stocks your agent focuses on**
   - Example: "VOLTAS.NS appears in 60% of latents"

2. **Temporal trading strategy**
   - Example: "Agent uses 7-15 day patterns (swing trading)"

3. **Critical features**
   - Example: "Volume signals appear in 21% of decisions"

4. **Decision predictability**
   - Example: "R²=0.53 means 53% of agent's logic is captured"

---

## ⚠️ Troubleshooting

### "File not found: data/custom.npz"
**Fix:** Your data file should already exist. Check:
```bash
dir data\custom.npz
```
If missing, collect PPO traces:
```bash
python collect_traces.py
```

### "Low alignment R² (<0.3)"
**Fixes:**
1. Increase capacity: `latent_dim=256, k=64`
2. Collect more data: `collect_traces.py` with more episodes
3. Train longer: `epochs=200`

### "Too many dead latents (>30%)"
**Fix:** Decrease sparsity:
```bash
# In train_ae.py, line 182
k = 48  # Was 32, now more latents active
```

### "Training too slow"
**Fix:** Reduce batch size:
```bash
# In train_ae.py, line 168
batch_size = 128  # Was 256
```

---

## 🎯 Validation Checklist

After running, verify:
- ✅ Reconstruction R² > 0.8 (model learns input structure)
- ✅ Alignment R² > 0.5 (latents predict PPO actions)
- ✅ Sparsity ~25% (32/128 active = efficient)
- ✅ Dead latents <20% (most latents useful)
- ✅ Features are interpretable (recognizable stock/lag patterns)

---

## 📚 Documentation

- **QUICKSTART.md**: General overview and examples
- **README.md**: Technical architecture details
- **model.py**: SAE implementation details (docstrings)

---

## 💡 Next Steps After Success

1. **Live trading integration**:
   - Use `checkpoints/large_run/sparse_ae_best.pt` to monitor which latents fire during live trades
   - Alert when critical latents (top 10) activate

2. **Portfolio optimization**:
   - Use discovered stock importance to adjust allocation limits
   - Example: If VOLTAS dominates, cap position size

3. **Risk management**:
   - Monitor latent activation patterns for regime changes
   - If temporal patterns shift, reduce position sizes

4. **Strategy refinement**:
   - Ablation study: remove features with low latent importance
   - Retrain PPO with focused feature set

---

## 📞 Support

If results don't meet expectations:
1. Check validation checklist above
2. Review `QUICKSTART.md` for examples
3. Inspect training logs in console output
4. Verify input data quality in `data/custom.npz`
