from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# TopK Sparse Autoencoder (based on "Scaling and Evaluating Sparse Autoencoders")
# =============================================================================

class TopKSparseAutoencoder(nn.Module):
    """
    k-Sparse Autoencoder following OpenAI's "Scaling and evaluating sparse autoencoders".
    
    Model Features
    1. TopK activation: Only keep top-k activations, directly controlling sparsity
    2. Pre-encoder bias: Centering input before encoding
    3. Tied/untied decoder weights option
    4. Unit-norm decoder columns (for interpretability)
    5. Auxiliary loss for dead latent prevention
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 512,
        num_hidden: int = 1,
        k: int = 5,
        tied_weights: bool = False,
        normalize_decoder: bool = True,
        use_pre_bias: bool = True,
        activation: Literal["relu", "gelu", "silu"] = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder
        
        # Pre-encoder bias (centers the input)
        self.use_pre_bias = use_pre_bias
        if use_pre_bias:
            self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Activation function
        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        act_cls = act_map.get(activation, nn.ReLU)
        
        # Encoder: input -> hidden layers -> latent (pre-activation)
        encoder_layers = []
        dim = input_dim
        for _ in range(num_hidden):
            encoder_layers.append(nn.Linear(dim, hidden_dim))
            encoder_layers.append(act_cls())
            dim = hidden_dim
        # Final encoder layer to latent space (no activation - TopK applied after)
        self.encoder_hidden = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()
        self.encoder_out = nn.Linear(dim, latent_dim)
        
        # Latent bias (applied before TopK)
        self.latent_bias = nn.Parameter(torch.zeros(latent_dim))
        
        # Decoder: latent -> hidden layers -> input
        if tied_weights and num_hidden == 0:
            # For single-layer tied weights, decoder weight = encoder weight transposed
            self.decoder_out = None
        else:
            decoder_layers = []
            dim = latent_dim
            for _ in range(num_hidden):
                decoder_layers.append(nn.Linear(dim, hidden_dim))
                decoder_layers.append(act_cls())
                dim = hidden_dim
            self.decoder_hidden = nn.Sequential(*decoder_layers) if decoder_layers else nn.Identity()
            self.decoder_out = nn.Linear(dim, input_dim)
        
        # Decoder bias (separate from weights for tied case)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Track dead latents
        self.register_buffer("latent_activation_count", torch.zeros(latent_dim))
        self.register_buffer("total_samples", torch.tensor(0, dtype=torch.long))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training dynamics."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
        
        # Initialize latent bias to encourage initial activation
        nn.init.uniform_(self.latent_bias, -0.1, 0.1)
    
    def _normalize_decoder_weights(self):
        """Normalize decoder output columns to unit norm (for interpretability)."""
        if self.normalize_decoder and self.decoder_out is not None:
            with torch.no_grad():
                self.decoder_out.weight.data = F.normalize(self.decoder_out.weight.data, dim=0)
    
    def _topk_activation(self, pre_act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply TopK activation: keep only top-k values, zero the rest.
        Returns: (sparse_activation, mask)
        """
        batch_size = pre_act.shape[0]
        k = min(self.k, self.latent_dim)
        
        # Get top-k indices
        topk_values, topk_indices = torch.topk(pre_act, k, dim=-1)
        
        # Create sparse activation
        sparse_act = torch.zeros_like(pre_act)
        sparse_act.scatter_(-1, topk_indices, F.relu(topk_values))
        
        # Create mask for auxiliary loss
        mask = torch.zeros_like(pre_act, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        
        return sparse_act, mask
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to sparse latent representation.
        Returns: (sparse_latent, pre_activation, topk_mask)
        """
        # Apply pre-bias centering
        if self.use_pre_bias:
            x = x - self.pre_bias
        
        # Encode through hidden layers
        hidden = self.encoder_hidden(x)
        pre_act = self.encoder_out(hidden) + self.latent_bias
        
        # Apply TopK sparsity
        sparse_latent, mask = self._topk_activation(pre_act)
        
        return sparse_latent, pre_act, mask
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent to reconstruction."""
        if self.tied_weights and self.decoder_out is None:
            # Tied weights: use transposed encoder weights
            recon = F.linear(z, self.encoder_out.weight.t())
        else:
            hidden = self.decoder_hidden(z)
            recon = self.decoder_out(hidden)
        
        # Add decoder bias and pre-bias (to invert centering)
        recon = recon + self.decoder_bias
        if self.use_pre_bias:
            recon = recon + self.pre_bias
        
        return recon
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass.
        Returns: (reconstruction, sparse_latent, aux_info)
        """
        sparse_latent, pre_act, mask = self.encode(x)
        recon = self.decode(sparse_latent)
        
        # Track activations for dead latent monitoring (only during training)
        if self.training:
            with torch.no_grad():
                self.latent_activation_count += mask.sum(dim=0).float()
                self.total_samples += x.shape[0]
        
        aux_info = {
            "pre_act": pre_act,
            "mask": mask,
            "sparsity": mask.float().mean(),
        }
        
        return recon, sparse_latent, aux_info
    
    def get_dead_latent_ratio(self) -> float:
        """Return fraction of latents that never activated."""
        if self.total_samples == 0:
            return 0.0
        dead = (self.latent_activation_count == 0).sum().item()
        return dead / self.latent_dim
    
    def reset_activation_stats(self):
        """Reset activation tracking buffers."""
        self.latent_activation_count.zero_()
        self.total_samples.zero_()


# =============================================================================
# Original L1-Sparse Autoencoder (kept for backward compatibility)
# =============================================================================

class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder with linear bottleneck and L1 latent penalty."""

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 512, num_hidden: int = 2):
        super().__init__()
        encoder_layers = []
        dim = input_dim
        for _ in range(num_hidden):
            encoder_layers.append(nn.Linear(dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            dim = hidden_dim
        encoder_layers.append(nn.Linear(dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        dim = latent_dim
        for _ in range(num_hidden):
            decoder_layers.append(nn.Linear(dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            dim = hidden_dim
        decoder_layers.append(nn.Linear(dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# =============================================================================
# JumpReLU Sparse Autoencoder (alternative from Anthropic's approach)
# =============================================================================

class JumpReLUSparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with JumpReLU activation (learnable threshold).
    Alternative to TopK that learns per-latent thresholds.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 512,
        num_hidden: int = 1,
        init_threshold: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Pre-bias
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Encoder
        encoder_layers = []
        dim = input_dim
        for _ in range(num_hidden):
            encoder_layers.append(nn.Linear(dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            dim = hidden_dim
        self.encoder_hidden = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()
        self.encoder_out = nn.Linear(dim, latent_dim)
        
        # Learnable thresholds per latent
        self.thresholds = nn.Parameter(torch.full((latent_dim,), init_threshold))
        
        # Decoder
        decoder_layers = []
        dim = latent_dim
        for _ in range(num_hidden):
            decoder_layers.append(nn.Linear(dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            dim = hidden_dim
        self.decoder_hidden = nn.Sequential(*decoder_layers) if decoder_layers else nn.Identity()
        self.decoder_out = nn.Linear(dim, input_dim)
        
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
    
    def _jump_relu(self, x: torch.Tensor) -> torch.Tensor:
        """JumpReLU: ReLU with per-latent threshold."""
        # x - threshold, then ReLU, but only where x > threshold
        return F.relu(x) * (x > self.thresholds).float()
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_centered = x - self.pre_bias
        hidden = self.encoder_hidden(x_centered)
        pre_act = self.encoder_out(hidden)
        sparse_latent = self._jump_relu(pre_act)
        return sparse_latent, pre_act
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_hidden(z)
        recon = self.decoder_out(hidden) + self.decoder_bias + self.pre_bias
        return recon
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        sparse_latent, pre_act = self.encode(x)
        recon = self.decode(sparse_latent)
        
        sparsity = (sparse_latent > 0).float().mean()
        aux_info = {"pre_act": pre_act, "sparsity": sparsity, "thresholds": self.thresholds}
        
        return recon, sparse_latent, aux_info


# =============================================================================
# Loss Functions
# =============================================================================

def sparse_ae_loss(recon: torch.Tensor, target: torch.Tensor, latent: torch.Tensor, sparsity_weight: float) -> dict:
    """L1-based sparse loss for Sparse Autoencoder."""
    mse = F.mse_loss(recon, target)
    sparsity = torch.mean(torch.abs(latent))
    loss = mse + sparsity_weight * sparsity
    return {"loss": loss, "mse": mse.detach(), "sparsity": sparsity.detach()}


def topk_sae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    latent: torch.Tensor,
    aux_info: dict,
    aux_weight: float = 1e-3,
    use_auxiliary: bool = True,
) -> dict:
    """
    Loss for TopK Sparse Autoencoder.
    
    Components:
    1. MSE reconstruction loss
    2. Auxiliary loss: encourages dead latents to activate (prevents dead neurons)
    """
    # Main reconstruction loss
    mse = F.mse_loss(recon, target)
    
    # Auxiliary loss: penalize pre-activations that are positive but got masked out
    # This encourages dormant latents to eventually cross the TopK threshold
    aux_loss = torch.tensor(0.0, device=recon.device)
    if use_auxiliary and "pre_act" in aux_info and "mask" in aux_info:
        pre_act = aux_info["pre_act"]
        mask = aux_info["mask"]
        # For masked-out latents, penalize positive pre-activations (they "wanted" to fire)
        masked_out_positive = F.relu(pre_act) * (~mask).float()
        aux_loss = masked_out_positive.mean()
    
    loss = mse + aux_weight * aux_loss
    
    sparsity = aux_info.get("sparsity", torch.tensor(0.0))
    
    return {
        "loss": loss,
        "mse": mse.detach(),
        "aux_loss": aux_loss.detach() if isinstance(aux_loss, torch.Tensor) else aux_loss,
        "sparsity": sparsity.detach() if isinstance(sparsity, torch.Tensor) else sparsity,
    }


def variance_explained(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute R² (variance explained) for reconstruction quality."""
    ss_res = ((target - recon) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2


# =============================================================================
# Model Factory
# =============================================================================

def create_sparse_autoencoder(
    model_type: Literal["topk", "l1", "jumprelu"] = "topk",
    **kwargs,
) -> nn.Module:
    """Factory function to create sparse autoencoder by type."""
    if model_type == "topk":
        return TopKSparseAutoencoder(**kwargs)
    elif model_type == "l1":
        return SparseAutoencoder(**kwargs)
    elif model_type == "jumprelu":
        return JumpReLUSparseAutoencoder(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
