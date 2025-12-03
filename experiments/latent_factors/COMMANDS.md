# Quick Command Reference

## 🎯 Standard Workflow (Copy-Paste Ready)

### Step 1: Train
```bash
python train_ae.py --data-path data/custom.npz --output-dir checkpoints/large_run --model-type topk --latent-dim 128 --k 32 --epochs 100 --normalize-decoder --use-auxiliary --save-best
```

### Step 2: Align
```bash
python align_factors.py --data-path data/custom.npz --checkpoint checkpoints/large_run/sparse_ae_best.pt --output-dir analysis/alignment_large
```

### Step 3: Extract Top Features
```bash
python extract_features.py --checkpoint checkpoints/large_run/sparse_ae_best.pt --data-path data/custom.npz --output-dir analysis/feature_extraction_large --tickers-path ..\..\tickers.csv
```

```bash
python show_top_features.py --features-file analysis/feature_extraction_large/top_features.json --alignment-file analysis/alignment_large/alignment_metrics.json --max-per-ticker 3
```

---

## ⚡ Quick Test (50 epochs, ~5 min)

**Step 1:**
```bash
python train_ae.py --data-path data/custom.npz --output-dir checkpoints/test --model-type topk --latent-dim 128 --k 32 --epochs 50 --normalize-decoder --use-auxiliary --save-best
```

**Step 2:**
```bash
python align_factors.py --data-path data/custom.npz --checkpoint checkpoints/test/sparse_ae_best.pt --output-dir analysis/alignment_test
```

**Step 3:**
```bash
python extract_features.py --checkpoint checkpoints/test/sparse_ae_best.pt --data-path data/custom.npz --output-dir analysis/feature_extraction_test --tickers-path ..\..\tickers.csv
```

**Step 4:**
```bash
python show_top_features.py --features-file analysis/feature_extraction_test/top_features.json --alignment-file analysis/alignment_test/alignment_metrics.json --max-per-ticker 3
```

---

## 🚀 High Capacity (256 latents, best alignment)
```bash
python train_ae.py --data-path data/custom.npz --output-dir checkpoints/xlarge --model-type topk --latent-dim 256 --k 64 --epochs 150 --normalize-decoder --use-auxiliary --save-best

python align_factors.py --data-path data/custom.npz --checkpoint checkpoints/xlarge/sparse_ae_best.pt --output-dir analysis/alignment_xlarge

python extract_features.py --checkpoint checkpoints/xlarge/sparse_ae_best.pt --data-path data/custom.npz --output-dir analysis/feature_extraction_xlarge --tickers-path ..\..\tickers.csv

python generate_financial_signals.py --features-file analysis/feature_extraction_large/top_features.json --alignment-file analysis/alignment_large/alignment_metrics.json --tickers-path ..\..\tickers.csv --max-per-ticker 3
```

---

## 📋 Prerequisites

**Before running, ensure:**
1. Data exists: `data/custom.npz` ✅ (You already have this!)
   - Contains: 230 samples, 39,964-dim observations, 97-stock actions

2. Tickers file exists: `../../tickers.csv`
   - Contains your stock ticker names

3. Dependencies installed: `pip install torch numpy scikit-learn`

---

## ✅ Expected Results

After completion, check:
- **Model**: `checkpoints/large_run/sparse_ae_best.pt`
- **Alignment**: `analysis/alignment_large/alignment_metrics.json`
  - Should show R² > 0.5 for excellent performance
- **Features**: `analysis/feature_extraction_large/top_features.json`
  - Maps latents to interpretable stock/time features

---

See **RUN_INSTRUCTIONS.md** for detailed explanations and troubleshooting.
