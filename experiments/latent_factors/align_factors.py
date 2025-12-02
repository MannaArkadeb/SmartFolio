#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TraceDataset
from .model import SparseAutoencoder, TopKSparseAutoencoder, JumpReLUSparseAutoencoder
from preproc import PreprocessConfig, preprocess_obs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align latent factors to PPO logits with a linear head")
    parser.add_argument("--data-path", required=True, help="Path to traces NPZ")
    parser.add_argument("--checkpoint", required=True, help="Path to sparse_ae.pt")
    parser.add_argument("--output-dir", default="experiments/latent_factors/analysis", help="Where to store outputs")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3, help="Ridge regularization strength")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    return parser.parse_args(argv)


def load_autoencoder(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    input_dim = ckpt.get("input_dim")
    latent_dim = config.get("latent_dim")
    hidden_dim = config.get("hidden_dim", 512)
    num_hidden = config.get("num_hidden", 2)
    model_type = config.get("model_type", "l1")
    
    if input_dim is None or latent_dim is None:
        raise ValueError("Checkpoint missing input_dim or latent_dim")
    
    # Create model based on type
    if model_type == "topk":
        k = config.get("k", 32)
        tied_weights = config.get("tied_weights", False)
        normalize_decoder = config.get("normalize_decoder", True)
        activation = config.get("activation", "relu")
        model = TopKSparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            k=k,
            tied_weights=tied_weights,
            normalize_decoder=normalize_decoder,
            activation=activation,
        )
    elif model_type == "jumprelu":
        model = JumpReLUSparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        )
    else:
        model = SparseAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        )
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    preproc_cfg = PreprocessConfig.from_dict(ckpt.get("preprocess"))
    meta = ckpt.get("meta")
    return model, {"config": config, "preprocess": preproc_cfg, "meta": meta, "model_type": model_type}


def fit_ridge(Z: np.ndarray, Y: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form ridge regression with bias term and numerical stability.
    
    Returns: (predictions, weights, active_mask)
    """
    # Convert to float64 first
    Z = Z.astype(np.float64)
    Y = Y.astype(np.float64)
    
    # Handle any inf/nan in input
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Identify active latent columns (non-zero variance)
    z_std = Z.std(axis=0)
    active_mask = z_std > 1e-6
    n_active = active_mask.sum()
    print(f"Active latents: {n_active}/{Z.shape[1]}")
    
    if n_active == 0:
        # No active latents, return zeros
        print("Warning: No active latents found!")
        return np.zeros_like(Y), np.zeros((Z.shape[1] + 1, Y.shape[1])), active_mask
    
    # Use only active columns
    Z_active = Z[:, active_mask]
    
    # Standardize active latents for numerical stability
    z_mean = Z_active.mean(axis=0, keepdims=True)
    z_std_active = Z_active.std(axis=0, keepdims=True) + 1e-8
    Z_norm = (Z_active - z_mean) / z_std_active
    
    ones = np.ones((Z_norm.shape[0], 1), dtype=np.float64)
    Zb = np.concatenate([Z_norm, ones], axis=1)
    
    dim = Zb.shape[1]
    A = Zb.T @ Zb + lam * np.eye(dim, dtype=np.float64)
    B = Zb.T @ Y
    
    try:
        W_active = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W_active = np.linalg.lstsq(A, B, rcond=None)[0]
    
    preds = Zb @ W_active
    
    # Reconstruct full weight matrix (with zeros for inactive latents)
    W_full = np.zeros((Z.shape[1] + 1, Y.shape[1]), dtype=np.float64)
    active_indices = np.where(active_mask)[0]
    W_full[active_indices, :] = W_active[:-1]  # Latent weights
    W_full[-1, :] = W_active[-1]  # Bias
    
    return preds, W_full, active_mask


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return r2


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TraceDataset(args.data_path, reshape=False)
    if dataset.logits is None:
        raise ValueError("Traces file does not contain 'logits'; cannot perform alignment.")

    device = torch.device(args.device)
    model, meta_cfg = load_autoencoder(Path(args.checkpoint).expanduser(), device)
    preproc_cfg: PreprocessConfig = meta_cfg["preprocess"]
    ckpt_meta = meta_cfg.get("meta") or {}
    model_type = meta_cfg.get("model_type", "l1")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    latents = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            obs_raw = batch["obs_flat"].to(device)
            # Use metadata from checkpoint if present, else dataset meta
            meta_for_preproc = ckpt_meta if ckpt_meta else dataset.meta
            obs = preprocess_obs(obs_raw, meta_for_preproc, preproc_cfg)
            
            # Handle different model return types
            output = model(obs)
            if model_type in ("topk", "jumprelu"):
                # Returns (recon, latent, aux_info)
                z = output[1]
            else:
                # Returns (recon, latent)
                z = output[1]
            
            latents.append(z.cpu().numpy())
            targets.append(batch["logits"].numpy())
    Z = np.concatenate(latents, axis=0)
    Y = np.concatenate(targets, axis=0)

    preds, weights, active_mask = fit_ridge(Z, Y, args.ridge_lambda)
    r2 = r2_score(Y, preds)
    mse = np.mean((Y - preds) ** 2, axis=0)

    metrics = {
        "ridge_lambda": args.ridge_lambda,
        "r2_mean": float(np.mean(r2)),
        "r2_per_action": r2.tolist(),
        "mse_mean": float(np.mean(mse)),
        "mse_per_action": mse.tolist(),
        "num_samples": int(Z.shape[0]),
        "latent_dim": int(Z.shape[1]),
        "active_latents": int(active_mask.sum()),
    }
    metrics_path = output_dir / "alignment_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved alignment metrics to {metrics_path}")

    np.savez_compressed(
        output_dir / "alignment_outputs.npz",
        latents=Z,
        logits=Y,
        preds=preds,
        weights=weights,
    )
    print(f"Saved alignment outputs to {output_dir / 'alignment_outputs.npz'}")


if __name__ == "__main__":
    main()
