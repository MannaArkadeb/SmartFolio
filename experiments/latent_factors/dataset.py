from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TraceDataset(Dataset):
    """
    Dataset loader for collected PPO traces.
    
    Serves 'activations' (latent embeddings) as the primary input for SAE training.
    """
    def __init__(self, data_path: str | Path, reshape: bool = False):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Trace file not found: {self.data_path}")
        
        # Load npz data
        data = np.load(self.data_path)
        
        # Load metadata if available
        self.meta = {}
        meta_path = self.data_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.meta = json.load(f)

        # Prefer activations if they exist (new collection format)
        if "activations" in data:
            self.activations = data["activations"].astype(np.float32)
            # Flatten: [steps, num_stocks, hidden_dim] -> [steps * num_stocks, hidden_dim]
            # This allows the SAE to learn per-stock concepts
            self.activations_flat = self.activations.reshape(-1, self.activations.shape[-1])
            self.data_source = "activations"
        else:
            raise ValueError(
                "Dataset missing 'activations'. Please re-run collect_traces.py with the updated code "
                "to capture agent internal embeddings."
            )

        # Keep other fields for reference/alignment
        if "logits" in data:
            self.logits = data["logits"].astype(np.float32).reshape(-1) # Flattened logits [steps * num_stocks]
        else:
            self.logits = None

    def __len__(self) -> int:
        return self.activations_flat.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input": torch.from_numpy(self.activations_flat[idx]),
        }
        if self.logits is not None:
            # We wrap logits in a 1-element tensor so it aligns with standard collation
            item["logits"] = torch.tensor([self.logits[idx]], dtype=torch.float32)
        return item