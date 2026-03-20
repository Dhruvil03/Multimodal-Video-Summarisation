"""
Phase 2 — Projection & Temporal Alignment
Multimodal Video Summarisation Pipeline

Takes:
  - Visual features  [T_v, 1024]  from Phase 1 (CLIP ViT-L/14 pooler = 1024-d)
  - Speech features  [T_s, 1024] from Phase 3 (Wav2Vec2 / Whisper) — plug in later

Outputs:
  - Fused token sequence  [T_v + T_s, 512]  ready for the Conformer encoder
"""

import torch
import torch.nn as nn
import math
from typing import Optional


# ─────────────────────────────────────────────
# 1. Sinusoidal Temporal Positional Encoding
# ─────────────────────────────────────────────

class TemporalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal PE (Vaswani et al. 2017) applied over the time axis.

    Each position t gets a unique pattern of sin/cos values across d_model
    dimensions, so the Conformer can distinguish 'frame 3' from 'frame 17'
    even though their visual content might look identical.

    Shape: [1, max_len, d_model]  (broadcast over batch)
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                         # [L, D]
        position = torch.arange(max_len).unsqueeze(1).float()      # [L, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )                                                           # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)               # even dims
        pe[:, 1::2] = torch.cos(position * div_term)               # odd dims
        pe = pe.unsqueeze(0)                                        # [1, L, D]
        self.register_buffer("pe", pe)                             # not a param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            x + PE:  [B, T, D]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 2. Modality Projection Head
# ─────────────────────────────────────────────

class ModalityProjection(nn.Module):
    """
    Projects a single modality's features from its native dim → d_model (512).

    Architecture:  Linear → LayerNorm → ReLU → Dropout
    - LayerNorm stabilises training (features from different models have
      very different scales — CLIP norms ~1.0, Wav2Vec2 can be much larger).
    - ReLU adds a mild non-linearity so the projection isn't purely linear.
    - Dropout regularises, especially helpful with small video datasets.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim]
        Returns:
            [B, T, d_model]
        """
        return self.proj(x)


# ─────────────────────────────────────────────
# 3. Modality Type Embeddings  (optional but recommended)
# ─────────────────────────────────────────────

class ModalityTypeEmbedding(nn.Module):
    """
    Learnable token-type embeddings that tell the Conformer which tokens are
    visual vs speech — analogous to BERT's segment embeddings.

    Two learned vectors of size d_model:
      type_id = 0  →  visual token
      type_id = 1  →  speech token
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(2, d_model)

    def forward(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        """
        Args:
            x:           [B, T, D]
            modality_id: 0 = visual, 1 = speech
        Returns:
            x + type_embedding:  [B, T, D]
        """
        B, T, _ = x.shape
        type_ids = torch.full((B, T), modality_id, dtype=torch.long, device=x.device)
        return x + self.embedding(type_ids)


# ─────────────────────────────────────────────
# 4. Full Projection & Alignment Module
# ─────────────────────────────────────────────

class ProjectionAlignmentModule(nn.Module):
    """
    Combines both modality projections, positional encoding, and type embeddings
    into a single module that outputs the fused token sequence.

    Speech path is optional — pass speech_features=None to run video-only
    (e.g. during Phase 1 & 2 development before speech is ready).

    Input shapes:
        visual_features:  [B, T_v, 768]   (CLIP ViT-L/14)
        speech_features:  [B, T_s, 1024]  (Wav2Vec2 / Whisper) — optional

    Output shape:
        [B, T_v + T_s, 512]   (or [B, T_v, 512] if speech is None)
    """

    def __init__(
        self,
        visual_dim: int = 768,
        speech_dim: int = 1024,
        d_model: int = 512,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_type_embeddings: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_type_embeddings = use_type_embeddings

        # Projection heads — one per modality
        self.visual_proj = ModalityProjection(visual_dim, d_model, dropout)
        self.speech_proj = ModalityProjection(speech_dim, d_model, dropout)

        # Shared positional encoding (applied after projection, before concat)
        self.pos_encoding = TemporalPositionalEncoding(d_model, max_seq_len, dropout)

        # Optional type embeddings
        if use_type_embeddings:
            self.type_embed = ModalityTypeEmbedding(d_model)

    def forward(
        self,
        visual_features: torch.Tensor,
        speech_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_features: [B, T_v, 768]
            speech_features: [B, T_s, 1024]  or None

        Returns:
            fused:    [B, T_v + T_s, 512]  — ready for Conformer
            src_mask: [B, T_v + T_s]        — True = valid token (for padding)
        """
        # ── Visual branch ─────────────────────────────
        v = self.visual_proj(visual_features)              # [B, T_v, 512]
        v = self.pos_encoding(v)                           # + temporal PE

        if self.use_type_embeddings:
            v = self.type_embed(v, modality_id=0)          # + visual type embed

        tokens = [v]
        T_v = v.size(1)

        # ── Speech branch (plug-in later) ─────────────
        T_s = 0
        if speech_features is not None:
            s = self.speech_proj(speech_features)          # [B, T_s, 512]
            s = self.pos_encoding(s)                       # + temporal PE

            if self.use_type_embeddings:
                s = self.type_embed(s, modality_id=1)      # + speech type embed

            tokens.append(s)
            T_s = s.size(1)

        # ── Concatenate along sequence dim ─────────────
        fused = torch.cat(tokens, dim=1)                   # [B, T_v + T_s, 512]

        # Build a padding mask (all True = no padding for now;
        # extend this when you add variable-length batching)
        B = fused.size(0)
        src_mask = torch.ones(B, T_v + T_s, dtype=torch.bool, device=fused.device)

        return fused, src_mask

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"type_embeddings={self.use_type_embeddings}"
        )


# ─────────────────────────────────────────────
# 5. Integration helper  (connects Phase 1 → Phase 2)
# ─────────────────────────────────────────────

def build_projection_module(
    d_model: int = 512,
    dropout: float = 0.1,
) -> ProjectionAlignmentModule:
    """
    Factory function with the exact dims expected from Phase 1.
    Call this once during model initialisation.
    """
    return ProjectionAlignmentModule(
        visual_dim=1024,    # CLIP ViT-L/14 pooler output (confirmed 1024-d)
        speech_dim=1024,    # Wav2Vec2-large
        d_model=d_model,
        dropout=dropout,
        use_type_embeddings=True,
    )


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Phase 2 — Projection & Alignment smoke test ===\n")

    B, T_v, T_s = 2, 32, 48         # batch=2, 32 visual frames, 48 speech frames

    # Simulated Phase 1 output (ViT-L/14 pooler = 1024-d)
    visual = torch.randn(B, T_v, 1024)

    module = build_projection_module(d_model=512)
    print(module)

    # ── Video-only (current state) ─────────────
    fused_v, mask_v = module(visual)
    print(f"\nVideo-only output : {fused_v.shape}")   # [2, 32, 512]
    print(f"Mask shape        : {mask_v.shape}")

    # ── With speech (Phase 3 ready) ───────────
    speech = torch.randn(B, T_s, 1024)
    fused_all, mask_all = module(visual, speech_features=speech)
    print(f"\nWith speech output: {fused_all.shape}") # [2, 80, 512]
    print(f"Mask shape        : {mask_all.shape}")

    # Parameter count
    total = sum(p.numel() for p in module.parameters())
    print(f"\nProjection module params: {total:,}")
