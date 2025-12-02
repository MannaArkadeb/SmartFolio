#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Literal, Sequence

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, random_split

from dataset import TraceDataset
from model import (
    SparseAutoencoder,
    TopKSparseAutoencoder,
    JumpReLUSparseAutoencoder,
    sparse_ae_loss,
    topk_sae_loss,
    variance_explained,
    create_sparse_autoencoder,
)
from preproc import PreprocessConfig, preprocess_obs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sparse autoencoder on PPO traces")
    parser.add_argument("--data-path", required=True, help="Path to traces NPZ produced by collect_traces.py")
    parser.add_argument("--output-dir", default="experiments/latent_factors/checkpoints", help="Where to store checkpoints")
    
    # Model architecture
    parser.add_argument("--model-type", type=str, default="topk", choices=["topk", "l1", "jumprelu"],
                        help="Type of sparse autoencoder: topk (recommended), l1 (original), jumprelu")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimensionality (increase for complex data)")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden layer width")
    parser.add_argument("--num-hidden", type=int, default=1, help="Number of hidden layers (1 is often best for SAE)")
    parser.add_argument("--k", type=int, default=32, help="TopK: number of active latents per sample")
    parser.add_argument("--tied-weights", action="store_true", help="Use tied encoder-decoder weights")
    parser.add_argument("--normalize-decoder", action="store_true", default=True, help="Normalize decoder columns")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "silu"],
                        help="Activation function in hidden layers")
    
    # Loss and regularization
    parser.add_argument("--sparsity-weight", type=float, default=1e-3, help="L1 penalty weight (for l1 model)")
    parser.add_argument("--aux-weight", type=float, default=1e-3, help="Auxiliary loss weight for dead latent prevention")
    parser.add_argument("--use-auxiliary", action="store_true", default=True, help="Use auxiliary loss for TopK")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "onecycle"],
                        help="Learning rate scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs for scheduler")
    
    # Data
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--drop-prev-weights", action="store_true", help="Zero out prev_weights block during training")
    parser.add_argument("--adj-scale", type=float, default=1.0, help="Scale factor for adjacency block")
    parser.add_argument("--ts-scale", type=float, default=1.0, help="Scale factor for ts_features block")
    parser.add_argument(
        "--prev-scale",
        type=float,
        default=0.1,
        help="Scale factor for prev_weights block (ignored if --drop-prev-weights)",
    )
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Epochs between detailed logs")
    parser.add_argument("--save-best", action="store_true", default=True, help="Save best model by val R²")
    
    return parser.parse_args(argv)


def compute_r2_batch(model, loader, device, meta, preproc_cfg, model_type: str) -> dict:
    """Compute R² and other metrics over a data loader."""
    model.eval()
    all_targets = []
    all_recons = []
    all_latents = []
    total_mse = 0.0
    total_sparsity = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            obs = preprocess_obs(batch["obs_flat"].to(device), meta, preproc_cfg)
            
            if model_type == "topk" or model_type == "jumprelu":
                recon, latent, aux_info = model(obs)
                sparsity = aux_info.get("sparsity", 0.0)
            else:
                recon, latent = model(obs)
                sparsity = (latent != 0).float().mean()
            
            all_targets.append(obs.cpu())
            all_recons.append(recon.cpu())
            all_latents.append(latent.cpu())
            total_mse += float(torch.nn.functional.mse_loss(recon, obs).item()) * obs.size(0)
            total_sparsity += float(sparsity if isinstance(sparsity, float) else sparsity.item()) * obs.size(0)
            count += obs.size(0)
    
    targets = torch.cat(all_targets, dim=0)
    recons = torch.cat(all_recons, dim=0)
    latents = torch.cat(all_latents, dim=0)
    
    # R² computation
    r2 = variance_explained(recons, targets).item()
    
    # Per-dimension R² (useful for analysis)
    ss_res = ((targets - recons) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    r2_per_dim = (1 - ss_res / (ss_tot + 1e-8)).mean().item()
    
    # Latent statistics
    latent_active = (latents.abs() > 1e-6).float().mean().item()
    latent_var = latents.var(dim=0).mean().item()
    
    return {
        "r2": r2,
        "r2_per_dim": r2_per_dim,
        "mse": total_mse / count,
        "sparsity": total_sparsity / count,
        "latent_active_ratio": latent_active,
        "latent_var": latent_var,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TraceDataset(args.data_path, reshape=False)
    input_dim = dataset.obs.shape[1]
    preproc_cfg = PreprocessConfig(
        drop_prev=args.drop_prev_weights,
        adj_scale=args.adj_scale,
        ts_scale=args.ts_scale,
        prev_scale=args.prev_scale,
    )

    total_len = len(dataset)
    val_size = max(1, int(total_len * args.val_split))
    train_size = total_len - val_size
    if train_size <= 0:
        train_size = 1
        val_size = max(0, total_len - train_size)
    
    # Use fixed seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    
    # Create model based on type
    if args.model_type == "topk":
        model = TopKSparseAutoencoder(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden,
            k=args.k,
            tied_weights=args.tied_weights,
            normalize_decoder=args.normalize_decoder,
            activation=args.activation,
        ).to(device)
    elif args.model_type == "jumprelu":
        model = JumpReLUSparseAutoencoder(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden,
        ).to(device)
    else:
        model = SparseAutoencoder(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden,
        ).to(device)
    
    print(f"Model type: {args.model_type}")
    print(f"Input dim: {input_dim}, Latent dim: {args.latent_dim}, Hidden: {args.hidden_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, args.epochs // 3), T_mult=2)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=args.warmup_epochs / args.epochs,
        )

    history = {
        "train_loss": [], "train_mse": [], "train_r2": [],
        "val_loss": [], "val_mse": [], "val_r2": [], "val_sparsity": [],
        "dead_latent_ratio": [], "lr": [],
    }
    
    best_val_r2 = -float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "reset_activation_stats"):
            model.reset_activation_stats()
        
        train_loss = 0.0
        train_mse = 0.0
        
        for batch in train_loader:
            obs = preprocess_obs(batch["obs_flat"].to(device), dataset.meta, preproc_cfg)
            optimizer.zero_grad()
            
            if args.model_type == "topk" or args.model_type == "jumprelu":
                recon, latent, aux_info = model(obs)
                losses = topk_sae_loss(
                    recon, obs, latent, aux_info,
                    aux_weight=args.aux_weight,
                    use_auxiliary=args.use_auxiliary,
                )
            else:
                recon, latent = model(obs)
                losses = sparse_ae_loss(recon, obs, latent, args.sparsity_weight)
            
            losses["loss"].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if args.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()
            
            train_loss += float(losses["loss"].item()) * obs.size(0)
            train_mse += float(losses["mse"].item()) * obs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_mse /= len(train_loader.dataset)
        
        if args.scheduler == "cosine" and scheduler is not None:
            scheduler.step()
        
        # Validation metrics
        val_metrics = compute_r2_batch(model, val_loader, device, dataset.meta, preproc_cfg, args.model_type)
        train_metrics = compute_r2_batch(model, train_loader, device, dataset.meta, preproc_cfg, args.model_type)
        
        # Dead latent tracking
        dead_ratio = 0.0
        if hasattr(model, "get_dead_latent_ratio"):
            dead_ratio = model.get_dead_latent_ratio()
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        history["train_loss"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["train_r2"].append(train_metrics["r2"])
        history["val_loss"].append(val_metrics["mse"])  # Using MSE as val_loss for TopK
        history["val_mse"].append(val_metrics["mse"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_sparsity"].append(val_metrics["sparsity"])
        history["dead_latent_ratio"].append(dead_ratio)
        history["lr"].append(current_lr)
        
        # Save best model
        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            best_epoch = epoch
            if args.save_best:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": vars(args),
                        "input_dim": input_dim,
                        "preprocess": preproc_cfg.to_dict(),
                        "meta": dataset.meta,
                        "epoch": epoch,
                        "val_r2": best_val_r2,
                    },
                    output_dir / "sparse_ae_best.pt",
                )
        
        # Logging
        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f} train_R²={train_metrics['r2']:.4f} | "
                f"val_R²={val_metrics['r2']:.4f} val_mse={val_metrics['mse']:.6f} | "
                f"sparsity={val_metrics['sparsity']:.3f} dead={dead_ratio:.3f} | "
                f"lr={current_lr:.2e}"
            )

    print(f"\nBest val R²: {best_val_r2:.4f} at epoch {best_epoch}")

    # Save final checkpoint
    ckpt_path = output_dir / "sparse_ae.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "input_dim": input_dim,
            "preprocess": preproc_cfg.to_dict(),
            "meta": dataset.meta,
            "epoch": args.epochs,
            "best_val_r2": best_val_r2,
        },
        ckpt_path,
    )
    print(f"Saved final checkpoint to {ckpt_path}")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Export latent codes for validation split
    latents: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            obs = preprocess_obs(batch["obs_flat"].to(device), dataset.meta, preproc_cfg)
            if args.model_type == "topk" or args.model_type == "jumprelu":
                _, latent, _ = model(obs)
            else:
                _, latent = model(obs)
            latents.append(latent.cpu().numpy())
            if "logits" in batch:
                targets.append(batch["logits"].cpu().numpy())
    if latents:
        latents_arr = np.concatenate(latents, axis=0)
        export = {"latents": latents_arr}
        if targets:
            export["logits"] = np.concatenate(targets, axis=0)
        np.savez_compressed(output_dir / "val_latents.npz", **export)
        print(f"Saved validation latents to {output_dir / 'val_latents.npz'}")


if __name__ == "__main__":
    main()
