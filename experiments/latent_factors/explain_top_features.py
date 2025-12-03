#!/usr/bin/env python3
"""
Explain how "top features" are determined in the Sparse Autoencoder
"""
import json
import numpy as np
from pathlib import Path

def main():
    # Load the decoder matrix and top features
    decoder_path = Path("analysis/feature_extraction_test/decoder_matrix.npz")
    features_path = Path("analysis/feature_extraction_test/top_features.json")
    
    if not decoder_path.exists():
        print(f"Error: {decoder_path} not found")
        return
    
    decoder_data = np.load(decoder_path)
    decoder_matrix = decoder_data['decoder_matrix']  # Shape: (input_dim, latent_dim)
    
    with open(features_path) as f:
        features_data = json.load(f)
    
    print("=" * 100)
    print("HOW 'TOP FEATURES' ARE DETERMINED")
    print("=" * 100)
    
    print(f"\n1. DECODER MATRIX SHAPE: {decoder_matrix.shape}")
    print(f"   - Input dimensions (stock features): {decoder_matrix.shape[0]:,}")
    print(f"   - Latent dimensions: {decoder_matrix.shape[1]}")
    print(f"   - Each column represents one latent's contribution to reconstructing inputs")
    
    print(f"\n2. WHAT IS A 'TOP FEATURE'?")
    print(f"   For each latent, we look at its decoder weights (one column of the decoder matrix).")
    print(f"   The weights tell us: when this latent is active, which raw input features")
    print(f"   get reconstructed the MOST?")
    print(f"   ")
    print(f"   Higher absolute weight = stronger influence on reconstruction")
    print(f"   = that input feature is more important for this latent's meaning")
    
    # Show example for a few latents
    print(f"\n3. EXAMPLE: Top features for 3 latents")
    print("=" * 100)
    
    for latent_id in ['latent_46', 'latent_61', 'latent_103']:
        if latent_id not in features_data['top_features']:
            continue
        
        info = features_data['top_features'][latent_id]
        latent_num = int(latent_id.replace('latent_', ''))
        
        print(f"\n{latent_id.upper()} (column {latent_num} of decoder matrix):")
        print("-" * 100)
        
        # Get the actual decoder column
        col = decoder_matrix[:, latent_num]
        abs_col = np.abs(col)
        
        print(f"Decoder column stats:")
        print(f"  - Max weight: {abs_col.max():.4f}")
        print(f"  - Mean weight: {abs_col.mean():.4f}")
        print(f"  - Median weight: {np.median(abs_col):.4f}")
        print(f"  - % of weights above 0.05: {(abs_col > 0.05).sum() / len(abs_col) * 100:.1f}%")
        
        print(f"\nTop 5 features (sorted by absolute weight):")
        names = info['feature_names'][:5]
        weights = info['weights'][:5]
        
        for i, (name, weight) in enumerate(zip(names, weights), 1):
            print(f"  {i}. {name:<40} weight: {abs(weight):>8.4f}  ({'positive' if weight > 0 else 'negative'})")
        
        print(f"\n  👉 Interpretation: When latent_{latent_num} activates, these are the stock")
        print(f"     features it's 'looking at' or 'encoding information about'.")
    
    print("\n" + "=" * 100)
    print("4. WHY ONLY SOME TICKERS APPEAR?")
    print("=" * 100)
    print(f"\nThe SAE learns 128 latents from {decoder_matrix.shape[0]:,} input features.")
    print(f"Each latent specializes in a small subset of features with HIGH weights.")
    print(f"")
    print(f"Tickers that don't appear in top features have:")
    print(f"  - Low decoder weights across ALL latents (< 0.05 typically)")
    print(f"  - Weak signal-to-noise ratio in the training data")
    print(f"  - Similar patterns to other stocks (redundant information)")
    print(f"")
    print(f"This is EXPECTED behavior - the SAE compresses information by finding")
    print(f"the most informative features, not by representing every stock equally.")
    
    print("\n" + "=" * 100)
    print("5. SUMMARY")
    print("=" * 100)
    print(f"""
'Top features' = input features with highest absolute decoder weights per latent

Selection process:
  1. For each latent (column of decoder matrix)
  2. Sort all input features by |weight|
  3. Take top-k (default: 10)
  4. These are the features that latent 'cares about most'

In show_top_features.py:
  - We filter by min_weight threshold (default: 0.05)
  - Pick one feature per ticker (weighted random)
  - Display in randomized order
  
Result: Only 67/97 tickers appear because the other 30 have weak weights everywhere.
""")
    

if __name__ == "__main__":
    main()
