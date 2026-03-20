"""
Phase 3 — Conformer Encoder
Multimodal Video Summarisation Pipeline

Takes:  fused sequence  [B, T, 512]  from Phase 2
Does:   N stacked Conformer blocks — each combines:
          - Multi-head self-attention  (global temporal context)
          - Depthwise convolution      (local temporal patterns)
          - Two feed-forward modules   (feature transformation)
Returns: contextualised sequence  [B, T, 512]  for the summarisation head

Reference: Gulati et al. "Conformer: Convolution-augmented Transformer
           for Speech Recognition" (2020) — https://arxiv.org/abs/2005.08100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ─────────────────────────────────────────────
# 1. Feed Forward Module
# ─────────────────────────────────────────────

class FeedForwardModule(nn.Module):
    """
    Pre-norm feed-forward block with half-step residual.

    Structure:
        x → LayerNorm → Linear(D→4D) → Swish → Dropout
          → Linear(4D→D) → Dropout → × 0.5 → + x

    The 0.5 scaling on the residual is specific to Conformer
    (unlike standard Transformers which use full residuals).
    This halved contribution lets the two FF modules
    (one before attention, one after conv) each contribute
    equally without dominating the block output.
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.linear2 = nn.Linear(d_model * expansion, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.SiLU()   # Swish activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return residual + 0.5 * x   # half-step residual


# ─────────────────────────────────────────────
# 2. Multi-Head Self-Attention Module
# ─────────────────────────────────────────────

class MultiHeadSelfAttentionModule(nn.Module):
    """
    Pre-norm multi-head self-attention with full residual.

    Uses PyTorch's built-in F.scaled_dot_product_attention which
    automatically uses Flash Attention when available on your GPU —
    important for longer sequences (38+ tokens from Phase 2).

    src_key_padding_mask: True = VALID token (our convention).
    PyTorch expects True = IGNORE, so we invert before passing in.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.norm    = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,    # [B, T, D] convention throughout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        [B, T, D]
            src_mask: [B, T]  True = valid token (inverted before MHSA)
        """
        residual = x
        x = self.norm(x)

        # Invert mask: PyTorch MHSA uses True = ignore padding
        key_padding_mask = ~src_mask if src_mask is not None else None

        x, _ = self.attn(
            query=x, key=x, value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return residual + self.dropout(x)


# ─────────────────────────────────────────────
# 3. Convolution Module
# ─────────────────────────────────────────────

class ConvolutionModule(nn.Module):
    """
    Depthwise separable convolution block — the key ingredient that
    distinguishes Conformer from a standard Transformer.

    Structure:
        x → LayerNorm
          → pointwise conv (D → 2D, GLU gate)
          → depthwise conv  (kernel_size, captures local patterns)
          → BatchNorm → Swish
          → pointwise conv (D → D)
          → Dropout → + x  (full residual)

    GLU (Gated Linear Unit): splits the 2D channels into two halves,
    uses one half to gate the other → σ(x₁) ⊙ x₂. Adds selectivity
    over which features flow through.

    kernel_size=31 is standard for speech/video — covers ~60s of context
    at 2s per token. Reduce to 15 if memory is tight.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2

        self.norm        = nn.LayerNorm(d_model)
        self.pointwise1  = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu         = nn.GLU(dim=1)           # halves channel dim: 2D→D
        self.depthwise   = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,                        # depthwise = one filter/channel
        )
        self.batch_norm  = nn.BatchNorm1d(d_model)
        self.act         = nn.SiLU()
        self.pointwise2  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args/Returns: [B, T, D]"""
        residual = x
        x = self.norm(x)

        # Conv1d expects [B, D, T]
        x = x.transpose(1, 2)
        x = self.pointwise1(x)    # [B, 2D, T]
        x = self.glu(x)           # [B,  D, T]
        x = self.depthwise(x)     # [B,  D, T]  local temporal patterns
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.pointwise2(x)    # [B,  D, T]
        x = self.dropout(x)

        # Back to [B, T, D]
        x = x.transpose(1, 2)
        return residual + x


# ─────────────────────────────────────────────
# 4. Single Conformer Block
# ─────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """
    One complete Conformer block:
        FF (½) → MHSA → Conv → FF (½) → LayerNorm

    Stacking N of these gives the full encoder.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1    = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attn   = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv   = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2    = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm   = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attn(x, src_mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


# ─────────────────────────────────────────────
# 5. Full Conformer Encoder (N stacked blocks)
# ─────────────────────────────────────────────

class ConformerEncoder(nn.Module):
    """
    Stack of N ConformerBlocks.

    Input:  [B, T, d_model]  — fused multimodal sequence from Phase 2
    Output: [B, T, d_model]  — temporally contextualised representations

    Recommended config for video summarisation:
        num_layers=4, d_model=512, num_heads=8  →  ~24M params in encoder
        num_layers=6, d_model=512, num_heads=8  →  ~36M params  (if GPU allows)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers  = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_expansion=ff_expansion,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        [B, T, d_model]  from ProjectionAlignmentModule
            src_mask: [B, T]           True = valid token

        Returns:
            [B, T, d_model]  contextualised sequence
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"num_layers={len(self.layers)}"
        )


# ─────────────────────────────────────────────
# 6. Factory
# ─────────────────────────────────────────────

def build_conformer_encoder(
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
) -> ConformerEncoder:
    """Build with the recommended defaults for video summarisation."""
    return ConformerEncoder(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_expansion=4,
        conv_kernel_size=31,
        dropout=dropout,
    )


# ─────────────────────────────────────────────
# 7. End-to-end pipeline test (Phase 1+1b+2+3)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from projection_alignment import build_projection_module

    print("=== Phase 3 — Conformer Encoder smoke test ===\n")

    # ── Option A: real video ──────────────────
    if len(sys.argv) > 1:
        from speech_feature_extractor import extract_all_features

        feats   = extract_all_features(sys.argv[1])
        visual  = feats["visual"].unsqueeze(0)     # [1, T_v, 1024]
        speech  = feats["speech"].unsqueeze(0)     # [1, T_s, 1024]

    # ── Option B: synthetic tensors ──────────
    else:
        print("No video path given — using synthetic tensors.\n")
        B, T_v, T_s = 2, 32, 6
        visual = torch.randn(B, T_v, 1024)
        speech = torch.randn(B, T_s, 1024)

    # Phase 2 — projection
    proj_module = build_projection_module(d_model=512)
    fused, mask = proj_module(visual, speech)
    print(f"Phase 2 output : {fused.shape}")       # [B, T_v+T_s, 512]

    # Phase 3 — Conformer
    encoder = build_conformer_encoder(
        d_model=512,
        num_layers=4,
        num_heads=8,
    )
    encoded = encoder(fused, src_mask=mask)
    print(f"Phase 3 output : {encoded.shape}")     # [B, T_v+T_s, 512]

    # Parameter counts
    proj_params = sum(p.numel() for p in proj_module.parameters())
    enc_params  = sum(p.numel() for p in encoder.parameters())
    print(f"\nProjection params : {proj_params:>10,}")
    print(f"Conformer  params : {enc_params:>10,}")
    print(f"Total (Ph2+Ph3)   : {proj_params + enc_params:>10,}")
    print("\nPhases 1 → 3 complete. Ready for summarisation head.")
