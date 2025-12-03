#!/usr/bin/env python3
"""
Extract Top Latent Features - Clean Output
Shows which stock features are most important, balanced across tickers
"""
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features-file", required=True, help="Path to top_features.json")
    p.add_argument("--alignment-file", required=True, help="Path to alignment_metrics.json")
    p.add_argument("--max-per-ticker", type=int, default=3, help="Max features per ticker")
    p.add_argument("--min-weight", type=float, default=0.05, help="Minimum feature weight")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Load data
    with open(args.features_file) as f:
        features_data = json.load(f)
    with open(args.alignment_file) as f:
        alignment_data = json.load(f)
    
    top_features = features_data.get('top_features', {})
    
    print("=" * 100)
    print("SPARSE AUTOENCODER - TOP LATENT FEATURES")
    print("=" * 100)
    
    print(f"\nModel Performance: R² = {alignment_data.get('r2_mean', 0):.4f}")
    print(f"Active Latents: {alignment_data.get('active_latents', 0)}/{len(top_features)}")
    
    # Collect all features with their weights
    all_features = []
    ticker_features = defaultdict(list)
    
    for latent_id, info in top_features.items():
        latent_num = int(latent_id.replace('latent_', ''))
        names = info.get('feature_names', [])
        weights = info.get('weights', [])
        
        # Iterate over all reported features for this latent (do not limit to top-5)
        for name, weight in zip(names, weights):
            if abs(weight) < args.min_weight:
                continue
            
            # Parse feature
            if '_t-' in name:
                parts = name.split('_')
                ticker = parts[0]
                lag = parts[1] if len(parts) > 1 else ''
                feature = parts[2] if len(parts) > 2 else ''
                
                entry = {
                    'latent': latent_num,
                    'ticker': ticker,
                    'lag': lag,
                    'feature': feature,
                    'weight': abs(weight),
                    'full_name': name
                }
                
                all_features.append(entry)
                ticker_features[ticker].append(entry)
    
    # Collect features up to max-per-ticker limit
    balanced_features = []
    ticker_counts = defaultdict(int)
    
    # Sort all features by weight first
    all_features.sort(key=lambda x: x['weight'], reverse=True)
    
    # Select features, limiting per ticker
    for feat in all_features:
        if ticker_counts[feat['ticker']] < args.max_per_ticker:
            balanced_features.append(feat)
            ticker_counts[feat['ticker']] += 1
    
    # Display
    print(f"\n\nTOP {len(balanced_features)} FEATURES (Max {args.max_per_ticker} per ticker, sorted by weight):")
    print("=" * 100)
    print(f"{'Rank':<6} {'Ticker':<15} {'Feature':<12} {'Time Lag':<12} {'Weight':<10} {'Latent':<8}")
    print("-" * 100)
    
    for i, feat in enumerate(balanced_features[:50], 1):
        print(f"{i:<6} {feat['ticker']:<15} {feat['feature']:<12} {feat['lag']:<12} "
              f"{feat['weight']:<10.4f} {feat['latent']:<8}")
    
    # Ticker distribution
    print(f"\n\nTICKER DISTRIBUTION:")
    print("=" * 100)
    ticker_counts = defaultdict(int)
    for feat in balanced_features:
        ticker_counts[feat['ticker']] += 1
    
    for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ticker:<20} {count:3d} features")
    
    print(f"\nTotal Unique Tickers: {len(ticker_counts)}")
    print("=" * 100)
    
    # Save
    output = {
        'model_r2': alignment_data.get('r2_mean', 0),
        'total_features': len(balanced_features),
        'unique_tickers': len(ticker_counts),
        'features': balanced_features[:50]
    }
    
    output_path = Path(args.features_file).parent / "top_latent_features.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Saved to: {output_path}\n")


if __name__ == "__main__":
    main()
