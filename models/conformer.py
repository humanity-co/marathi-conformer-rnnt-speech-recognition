"""
Conformer Encoder — Lightweight Implementation for Marathi ASR

Architecture (per block):
    x → FFN/2 → MHSA → ConvModule → FFN/2 → LayerNorm → out

The Conformer combines:
  • Self-attention for capturing global/long-range dependencies in speech
  • Depthwise convolution for local feature extraction (phonetic patterns)
  • Half-step feed-forward residuals (Macaron-Net style)

This produces state-of-the-art speech representations while being
computationally lighter than full Transformer encoders.

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
for Speech Recognition", 2020.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Activation ────────────────────────────────────────────────────────────────

class Swish(nn.Module):
    """Swish activation: x * sigmoid(x). Smoother than ReLU."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


# ── Positional Encoding ──────────────────────────────────────────────────────

class RelativePositionalEncoding(nn.Module):
    """
    Sinusoidal relative positional encoding.
    Generates position encodings up to max_len, used by attention layers
    to incorporate sequence order information without absolute positions.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: (B, T, D)"""
        return x + self.pe[:, : x.size(1)]


# ── Conv Subsampling ─────────────────────────────────────────────────────────

class Conv2dSubsampling(nn.Module):
    """
    Two-layer Conv2D subsampling that reduces the time dimension by a
    factor of 4 (stride 2 in each layer). This dramatically reduces
    sequence length before the expensive self-attention layers.

    Input:  (B, T, n_mels)  — mel spectrogram
    Output: (B, T//4, d_model)
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After two stride-2 convolutions, freq dim = ceil(input_dim / 4)
        subsampled_freq = math.ceil(input_dim / 2)
        subsampled_freq = math.ceil(subsampled_freq / 2)
        self.linear = nn.Linear(d_model * subsampled_freq, d_model)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, n_mels)
            lengths: (B,) original time lengths

        Returns:
            out: (B, T', d_model)   where T' ≈ T // 4
            new_lengths: (B,)
        """
        x = x.unsqueeze(1)  # (B, 1, T, n_mels) — treat as single-channel image
        x = self.conv(x)    # (B, d_model, T', freq')
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.linear(x)  # (B, T', d_model)

        # Update lengths: each stride-2 conv does ceil division
        new_lengths = ((lengths - 1) // 2 + 1)
        new_lengths = ((new_lengths - 1) // 2 + 1)
        return x, new_lengths


# ── Feed-Forward Module ──────────────────────────────────────────────────────

class FeedForwardModule(nn.Module):
    """
    Position-wise Feed-Forward with Swish activation.
    FFN(x) = Dropout(Linear(Dropout(Swish(Linear(LayerNorm(x))))))
    """
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = d_model * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(d_model, inner_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Convolution Module ───────────────────────────────────────────────────────

class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module:
        Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout

    Uses depthwise separable convolution for efficient local feature extraction.
    The kernel_size controls the receptive field for local context.
    """
    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2

        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=padding, groups=d_model  # groups=d_model → depthwise
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x.transpose(1, 2)        # (B, D, T)
        x = self.pointwise_conv1(x)   # (B, 2D, T)
        x = self.glu(x)               # (B, D, T)
        x = self.depthwise_conv(x)    # (B, D, T)
        x = self.batch_norm(x)        # (B, D, T)
        x = self.activation(x)        # (B, D, T)
        x = self.pointwise_conv2(x)   # (B, D, T)
        x = self.dropout(x)           # (B, D, T)
        return x.transpose(1, 2)      # (B, T, D)


# ── Multi-Head Self-Attention ────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Pre-norm multi-head self-attention with positional encoding injection.
    Uses PyTorch's efficient nn.MultiheadAttention underneath.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) — True for padding positions
        """
        out, _ = self.mha(x, x, x, key_padding_mask=mask)
        return out


# ── Conformer Block ──────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """
    Single Conformer block (Macaron-Net style):

        x → LayerNorm → FFN × 0.5 + x
          → LayerNorm → MHSA + x
          → LayerNorm → ConvModule + x
          → LayerNorm → FFN × 0.5 + x
          → LayerNorm → out
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        ffn_expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_expansion_factor, dropout)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_expansion_factor, dropout)

        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.norm_mhsa = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 1. Half-step Feed-Forward
        x = x + 0.5 * self.ffn1(self.norm_ffn1(x))

        # 2. Multi-Head Self-Attention
        x = x + self.mhsa(self.norm_mhsa(x), mask=mask)

        # 3. Convolution Module
        x = x + self.conv(self.norm_conv(x))

        # 4. Half-step Feed-Forward
        x = x + 0.5 * self.ffn2(self.norm_ffn2(x))

        # 5. Final Layer Norm
        x = self.norm_final(x)
        return x


# ── Conformer Encoder ────────────────────────────────────────────────────────

class ConformerEncoder(nn.Module):
    """
    Full Conformer Encoder stack:
        Mel features → Conv2D subsampling (4x) → Positional Encoding
        → N × ConformerBlock → encoder output

    Args:
        input_dim: Number of mel filterbank channels (default 80)
        d_model: Hidden dimension throughout the encoder (default 256)
        num_blocks: Number of stacked conformer blocks (default 6)
        num_heads: Attention heads (default 4)
        conv_kernel_size: Depthwise conv kernel (default 15)
        dropout: Dropout probability (default 0.1)
    """
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        num_blocks: int = 6,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        ffn_expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsampling = Conv2dSubsampling(input_dim, d_model)
        self.pos_encoding = RelativePositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                ffn_expansion_factor=ffn_expansion_factor,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim) — log-mel spectrogram features
            lengths: (B,) — valid time lengths before padding

        Returns:
            x: (B, T', d_model) — encoded representations
            lengths: (B,) — updated lengths after subsampling
        """
        x, lengths = self.subsampling(x, lengths)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Build padding mask: True where positions are padding
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]

        for block in self.blocks:
            x = block(x, mask=mask)

        return x, lengths
