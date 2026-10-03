# ═══════════════════════════════════════════════════════════════════════════════
# model.py - UltimateECGHybrid (CPU-OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Optimizations:
# ✅ Gradient checkpointing (saves memory on CPU)
# ✅ Efficient parameter initialization
# ✅ NaN/Inf guards
# ✅ Optimized forward pass
# ✅ Memory-friendly attention hooks
#
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils import parametrizations

try:
    from config_OPTIMIZED import (
        DEVICE, LATENT_DIM, NUM_CLASSES, OUTPUT_WAVEFORM_LEN,
        NHEAD, NUM_TRANSFORMER_LAYERS, USE_GRADIENT_CHECKPOINTING, IS_CPU
    )
except ImportError:
    from config import (
        DEVICE, LATENT_DIM, NUM_CLASSES, OUTPUT_WAVEFORM_LEN,
        NHEAD, NUM_TRANSFORMER_LAYERS
    )
    USE_GRADIENT_CHECKPOINTING = IS_CPU = True

# ─────────────────────────────────────────────────────────────────────────────
# 1. DUAL ENCODER (CPU-Optimized)
# ─────────────────────────────────────────────────────────────────────────────

class DualEncoder(nn.Module):
    """
    Parallel ViT-B/16 and EfficientNet-B0 encoders.
    Optimized for CPU with reduced memory footprint.
    """
    
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        
        # ViT-B/16
        self.vit = models.vit_b_16(weights="DEFAULT")
        vit_out = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        
        # EfficientNet-B0
        self.eff = models.efficientnet_b0(weights="DEFAULT")
        eff_out = self.eff.classifier[1].in_features
        self.eff.classifier = nn.Identity()
        
        # Fusion (lightweight MLP)
        self.fusion = nn.Sequential(
            nn.Linear(vit_out + eff_out, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        
        # Attention hooks for XAI
        self._attn_weights = []
        self._hooks = []

    def _make_hook(self):
        """Create attention weight hook"""
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                self._attn_weights.append(out[1].detach().cpu())
        return hook

    def register_attention_hooks(self):
        """Register hooks on ViT attention layers"""
        self.remove_hooks()
        for block in self.vit.encoder.layers:
            h = block.self_attention.register_forward_hook(self._make_hook())
            self._hooks.append(h)

    def remove_hooks(self):
        """Clean up hooks (important for memory)"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory efficiency"""
        self._attn_weights.clear()
        
        # ViT forward
        vit_feat = self.vit(x)
        
        # EfficientNet forward
        eff_feat = self.eff(x)
        
        # Fusion
        fused = torch.cat([vit_feat, eff_feat], dim=1)
        output = self.fusion(fused)
        
        return output

# ─────────────────────────────────────────────────────────────────────────────
# 2. BIDIRECTIONAL CONVLSTM (Memory-Efficient)
# ─────────────────────────────────────────────────────────────────────────────

class BiConvLSTMCell(nn.Module):
    """Bidirectional ConvLSTM optimized for CPU"""
    
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        
        self.fwd = nn.Conv1d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=pad
        )
        self.bwd = nn.Conv1d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=pad
        )

    def _step(self, x_t, h, c, conv):
        """Single LSTM step"""
        gates = conv(torch.cat([x_t, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with bidirectional processing.
        CPU-optimized with minimal temporary allocations.
        """
        B, C, T = x.size()
        
        # Initialize states on input device
        device = x.device
        h_f = torch.zeros(B, self.hidden_channels, T, device=device, dtype=x.dtype)
        c_f = torch.zeros(B, self.hidden_channels, T, device=device, dtype=x.dtype)
        h_b = torch.zeros(B, self.hidden_channels, T, device=device, dtype=x.dtype)
        c_b = torch.zeros(B, self.hidden_channels, T, device=device, dtype=x.dtype)
        
        outs_f = []
        outs_b = []
        
        # Forward pass
        for t in range(T):
            x_t_expanded = x[:, :, t:t+1].expand(-1, -1, T)
            h_f, c_f = self._step(x_t_expanded, h_f, c_f, self.fwd)
            outs_f.append(h_f.mean(-1, keepdim=True))
        
        # Backward pass
        for t in reversed(range(T)):
            x_t_expanded = x[:, :, t:t+1].expand(-1, -1, T)
            h_b, c_b = self._step(x_t_expanded, h_b, c_b, self.bwd)
            outs_b.insert(0, h_b.mean(-1, keepdim=True))
        
        # Concatenate forward and backward
        fwd_out = torch.cat(outs_f, -1)
        bwd_out = torch.cat(outs_b, -1)
        
        return torch.cat([fwd_out, bwd_out], dim=1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. INFORMER BLOCK (Lightweight Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class InformerBlock(nn.Module):
    """Lightweight Transformer with learnable positional embeddings"""
    
    def __init__(self, d_model: int, nhead: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with positional encoding"""
        return self.transformer(x + self.pos[:, :x.size(1), :])

# ─────────────────────────────────────────────────────────────────────────────
# 4. TCN BLOCKS (Temporal Convolutional Network)
# ─────────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    """Single dilated TCN residual block with weight normalization"""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        
        # Modern weight normalization
        self.conv1 = parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        )
        self.conv2 = parametrizations.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        )
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection"""
        # First conv + activation
        o = self.drop(self.relu(self.conv1(x)[..., :x.size(-1)]))
        
        # Second conv + activation
        o = self.drop(self.relu(self.conv2(o)[..., :x.size(-1)]))
        
        # Residual connection
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(o + residual)

class TCNPredictor(nn.Module):
    """Stack of TCN blocks with upsampling to output length"""
    
    def __init__(self, in_channels: int, channels: list = None, 
                 kernel_size: int = 3, output_len: int = 500):
        super().__init__()
        channels = channels or [256, 128, 64]
        
        # Stack of TCN blocks with increasing dilation
        self.tcn = nn.Sequential(*[
            TCNBlock(
                in_channels if i == 0 else channels[i - 1],
                ch,
                kernel_size,
                2 ** i
            )
            for i, ch in enumerate(channels)
        ])
        
        self.head = nn.Conv1d(channels[-1], 1, 1)
        self.up = nn.Upsample(size=output_len, mode="linear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with upsampling"""
        tcn_out = self.tcn(x)
        head_out = self.head(tcn_out)
        return self.up(head_out).squeeze(1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL HYBRID MODEL (CPU-OPTIMIZED)
# ─────────────────────────────────────────────────────────────────────────────

class UltimateECGHybrid(nn.Module):
    """
    Complete ECG analysis model.
    
    Input: [B, 3, 224, 224] ECG image
    Output: 
        - waveform: [B, output_len] (predicted next waveform)
        - logits: [B, num_classes] (disease classification)
    """
    
    def __init__(self, latent_dim: int = 512, num_classes: int = 5, 
                 output_len: int = 500, nhead: int = 8, num_layers: int = 3):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_gradient_checkpointing = USE_GRADIENT_CHECKPOINTING
        
        # Encoder
        self.encoder = DualEncoder(latent_dim)
        
        # Sequence expansion
        self.seq_expand = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 16),
            nn.GELU(),
        )
        self.seq_len = 16
        
        # Sequential layers
        self.bilstm = BiConvLSTMCell(latent_dim, latent_dim // 2)
        self.informer = InformerBlock(latent_dim, nhead, num_layers)
        self.lstm_proj = nn.Linear(latent_dim, latent_dim)
        
        # Decoders
        self.tcn = TCNPredictor(latent_dim, [256, 128, 64], output_len=output_len)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass with gradient checkpointing for CPU efficiency.
        """
        B = x.size(0)
        
        # Encoder
        if self.use_gradient_checkpointing and self.training:
            latent = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        else:
            latent = self.encoder(x)
        
        # Expand to sequence
        seq = self.seq_expand(latent).view(B, self.latent_dim, self.seq_len)
        
        # BiConvLSTM
        if self.use_gradient_checkpointing and self.training:
            bilstm = torch.utils.checkpoint.checkpoint(self.bilstm, seq, use_reentrant=False)
        else:
            bilstm = self.bilstm(seq)
        
        bilstm = bilstm[:, :self.latent_dim, :]
        
        # Informer
        if self.use_gradient_checkpointing and self.training:
            trans = torch.utils.checkpoint.checkpoint(
                self.informer, 
                bilstm.permute(0, 2, 1), 
                use_reentrant=False
            )
        else:
            trans = self.informer(bilstm.permute(0, 2, 1))
        
        trans = self.lstm_proj(trans)
        
        # TCN (Waveform prediction)
        if self.use_gradient_checkpointing and self.training:
            wave = torch.utils.checkpoint.checkpoint(
                self.tcn,
                trans.permute(0, 2, 1),
                use_reentrant=False
            )
        else:
            wave = self.tcn(trans.permute(0, 2, 1))
        
        # Classifier (Disease classification)
        logits = self.classifier(trans.mean(dim=1))
        
        # NaN/Inf guard
        if torch.isnan(wave).any() or torch.isinf(wave).any():
            wave = torch.nan_to_num(wave, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return wave, logits

# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(weights_path: str = None, eval_mode: bool = True) -> UltimateECGHybrid:
    """
    Build and optionally load trained UltimateECGHybrid model.
    
    Args:
        weights_path: Path to pre-trained weights (.pth file)
        eval_mode: If True, set model to eval mode
    
    Returns:
        Model on DEVICE
    """
    model = UltimateECGHybrid(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        output_len=OUTPUT_WAVEFORM_LEN,
        nhead=NHEAD,
        num_layers=NUM_TRANSFORMER_LAYERS
    ).to(DEVICE)
    
    # Load weights if provided
    if weights_path:
        try:
            state = torch.load(weights_path, map_location=DEVICE)
            
            # Handle DataParallel-wrapped checkpoints
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            
            model.load_state_dict(state)
            print(f"✓ Weights loaded from: {weights_path}")
        
        except FileNotFoundError:
            print(f"⚠️  Weights file not found: {weights_path}")
        except RuntimeError as e:
            print(f"⚠️  Error loading weights: {e}")
    
    if eval_mode:
        model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"Model loaded successfully")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {DEVICE}")
    print(f"Gradient checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    print(f"{'='*70}\n")
    
    return model

# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running model smoke test...")
    
    model = build_model()
    dummy_input = torch.randn(2, 3, 224, 224).to(DEVICE)
    
    with torch.no_grad():
        waveform, logits = model(dummy_input)
    
    print(f"✓ Smoke test passed")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Waveform output: {waveform.shape}")
    print(f"  Classification output: {logits.shape}")
