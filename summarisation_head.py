"""
Phase 4 — Summarisation Head + Full End-to-End Model
Multimodal Video Summarisation Pipeline

Takes:  contextualised sequence  [B, T, 512]  from Phase 3 (Conformer)
Does:   cross-attention decoder auto-regressively generates summary tokens
Returns: summary token ids  [B, L]  → decoded text

Architecture:
    Conformer output → cross-attention decoder (GPT2-style) → text tokens

We use GPT-2 as the decoder backbone:
  - Pre-trained language model weights  →  strong language prior out of the box
  - Cross-attention layers added        →  attend over Conformer encoder output
  - Fine-tuned end-to-end              →  learns to ground language in video

Reference: "Video-LLaMA", "UniVTG", and related encoder-decoder video models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


# ─────────────────────────────────────────────
# 1. Cross-Attention Layer
#    (injected into GPT-2 decoder blocks)
# ─────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """
    Adds a cross-attention sublayer to a decoder block so it can
    attend over the Conformer's encoder output.

    Placed after GPT-2's self-attention and before its FFN:
        Self-Attn → Cross-Attn (new) → FFN

    encoder_dim: d_model from Conformer (512)
    decoder_dim: GPT-2 hidden size     (768 for gpt2-base)
    """

    def __init__(
        self,
        decoder_dim: int,
        encoder_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm    = nn.LayerNorm(decoder_dim)
        self.q_proj  = nn.Linear(decoder_dim, decoder_dim)
        self.k_proj  = nn.Linear(encoder_dim,  decoder_dim)
        self.v_proj  = nn.Linear(encoder_dim,  decoder_dim)
        self.out_proj = nn.Linear(decoder_dim, decoder_dim)
        self.attn    = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,               # [B, L, decoder_dim]  decoder hidden states
        encoder_out: torch.Tensor,      # [B, T, encoder_dim]  Conformer output
        encoder_mask: Optional[torch.Tensor] = None,   # [B, T]
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        q = self.q_proj(x)
        k = self.k_proj(encoder_out)
        v = self.v_proj(encoder_out)

        key_padding_mask = ~encoder_mask if encoder_mask is not None else None

        attn_out, _ = self.attn(
            query=q, key=k, value=v,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return residual + self.dropout(self.out_proj(attn_out))


# ─────────────────────────────────────────────
# 2. Encoder Projection
#    (aligns Conformer dim → GPT-2 dim)
# ─────────────────────────────────────────────

class EncoderProjection(nn.Module):
    """
    Projects Conformer output (512-d) into GPT-2's embedding space (768-d).
    Applied once before decoding — cheap linear alignment layer.
    """

    def __init__(self, encoder_dim: int = 512, decoder_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # [B, T, decoder_dim]


# ─────────────────────────────────────────────
# 3. Summarisation Head
#    (GPT-2 decoder with cross-attention)
# ─────────────────────────────────────────────

class SummarisationHead(nn.Module):
    """
    GPT-2 decoder augmented with cross-attention over the Conformer output.

    Training:   teacher-forced — ground truth summary tokens as decoder input
    Inference:  greedy / beam / nucleus sampling

    We inject one CrossAttentionLayer per GPT-2 transformer block,
    positioned between the self-attention and the feed-forward network.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        gpt2_variant: str = "gpt2",        # gpt2 | gpt2-medium | gpt2-large
        num_cross_attn_heads: int = 8,
        dropout: float = 0.1,
        max_summary_len: int = 150,
    ):
        super().__init__()
        self.max_summary_len = max_summary_len

        # Load GPT-2 tokeniser & model
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_variant)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_variant)
        decoder_dim = self.gpt2.config.n_embd   # 768 for gpt2-base

        # Project encoder output to GPT-2 dim
        self.enc_proj = EncoderProjection(encoder_dim, decoder_dim)

        # Inject cross-attention into every GPT-2 block
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(
                decoder_dim=decoder_dim,
                encoder_dim=decoder_dim,   # after enc_proj
                num_heads=num_cross_attn_heads,
                dropout=dropout,
            )
            for _ in range(self.gpt2.config.n_layer)
        ])

        # Vocab size
        self.vocab_size = self.gpt2.config.vocab_size

    def forward(
        self,
        encoder_out: torch.Tensor,          # [B, T, 512]  Conformer output
        encoder_mask: torch.Tensor,         # [B, T]
        target_ids: Optional[torch.Tensor] = None,  # [B, L]  teacher-forced
    ) -> dict:
        """
        Training mode (target_ids provided):
            Returns loss + logits for every position.

        Inference mode (target_ids=None):
            Returns generated token ids.
        """
        B = encoder_out.size(0)
        device = encoder_out.device

        # Project encoder output once
        enc = self.enc_proj(encoder_out)    # [B, T, 768]

        if target_ids is not None:
            return self._forward_train(enc, encoder_mask, target_ids)
        else:
            return self._forward_generate(enc, encoder_mask, device, B)

    def _forward_train(
        self,
        enc: torch.Tensor,
        encoder_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> dict:
        """Teacher-forced training pass."""
        # Shift targets: input = [BOS, t1, t2, ...], label = [t1, t2, ..., EOS]
        decoder_input = target_ids[:, :-1]
        labels        = target_ids[:, 1:].clone()

        # Mask padding in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        # GPT-2 self-attention forward (get hidden states layer by layer)
        hidden = self.gpt2.transformer.wte(decoder_input)   # token embeds
        hidden = hidden + self.gpt2.transformer.wpe(
            torch.arange(hidden.size(1), device=hidden.device)
        )                                                    # + position embeds

        for i, block in enumerate(self.gpt2.transformer.h):
            # GPT-2 self-attention
            hidden = block(hidden)[0]
            # Cross-attention over Conformer output
            hidden = self.cross_attn_layers[i](hidden, enc, encoder_mask)

        hidden = self.gpt2.transformer.ln_f(hidden)
        logits = self.gpt2.lm_head(hidden)       # [B, L-1, vocab]

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def _forward_generate(
        self,
        enc: torch.Tensor,
        encoder_mask: torch.Tensor,
        device: torch.device,
        B: int,
    ) -> dict:
        """Greedy decoding — generates one token at a time."""
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(self.max_summary_len):
            hidden = self.gpt2.transformer.wte(generated)
            hidden = hidden + self.gpt2.transformer.wpe(
                torch.arange(hidden.size(1), device=device)
            )

            for i, block in enumerate(self.gpt2.transformer.h):
                hidden = block(hidden)[0]
                hidden = self.cross_attn_layers[i](hidden, enc, encoder_mask)

            hidden = self.gpt2.transformer.ln_f(hidden)
            logits = self.gpt2.lm_head(hidden[:, -1, :])  # last token only
            next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            finished  = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        return {"generated_ids": generated}

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        """Convert token id tensors → human-readable strings."""
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in token_ids
        ]


# ─────────────────────────────────────────────
# 4. Full End-to-End Model
# ─────────────────────────────────────────────

class MultimodalVideoSummariser(nn.Module):
    """
    Complete end-to-end model combining all four phases:

        Phase 2: ProjectionAlignmentModule   →  [B, T, 512]
        Phase 3: ConformerEncoder            →  [B, T, 512]
        Phase 4: SummarisationHead           →  summary text

    Phase 1 / 1b (feature extraction) are run offline and cached.
    This module operates on pre-extracted features.
    """

    def __init__(
        self,
        # Phase 2
        visual_dim: int = 1024,
        speech_dim: int = 1024,
        d_model: int = 512,
        # Phase 3
        num_conformer_layers: int = 4,
        num_attn_heads: int = 8,
        conv_kernel_size: int = 31,
        # Phase 4
        gpt2_variant: str = "gpt2",
        max_summary_len: int = 150,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Phase 2
        from projection_alignment import ProjectionAlignmentModule
        self.projection = ProjectionAlignmentModule(
            visual_dim=visual_dim,
            speech_dim=speech_dim,
            d_model=d_model,
            dropout=dropout,
            use_type_embeddings=True,
        )

        # Phase 3
        from conformer_encoder import ConformerEncoder
        self.conformer = ConformerEncoder(
            d_model=d_model,
            num_layers=num_conformer_layers,
            num_heads=num_attn_heads,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        # Phase 4
        self.summariser = SummarisationHead(
            encoder_dim=d_model,
            gpt2_variant=gpt2_variant,
            dropout=dropout,
            max_summary_len=max_summary_len,
        )

    def forward(
        self,
        visual: torch.Tensor,                           # [B, T_v, 1024]
        speech: Optional[torch.Tensor] = None,          # [B, T_s, 1024]
        target_ids: Optional[torch.Tensor] = None,      # [B, L]  training only
    ) -> dict:
        # Phase 2 — project & align
        fused, mask = self.projection(visual, speech)   # [B, T, 512]

        # Phase 3 — contextualise with Conformer
        encoded = self.conformer(fused, src_mask=mask)  # [B, T, 512]

        # Phase 4 — generate summary
        return self.summariser(encoded, mask, target_ids)

    def summarise(
        self,
        visual: torch.Tensor,
        speech: Optional[torch.Tensor] = None,
    ) -> list[str]:
        """Convenience inference method — returns list of summary strings."""
        self.eval()
        with torch.no_grad():
            output = self.forward(visual, speech, target_ids=None)
        return self.summariser.decode(output["generated_ids"])

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "projection":  n(self.projection),
            "conformer":   n(self.conformer),
            "summariser":  n(self.summariser),
            "total":       n(self),
        }


# ─────────────────────────────────────────────
# 5. Smoke test — full pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== Phase 4 — Full End-to-End Model ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Build full model
    model = MultimodalVideoSummariser(
        visual_dim=1024,
        speech_dim=1024,
        d_model=512,
        num_conformer_layers=4,
        gpt2_variant="gpt2",
        max_summary_len=150,
    ).to(device)

    # ── Option A: real video ──────────────────
    if len(sys.argv) > 1:
        from speech_feature_extractor import extract_all_features

        feats  = extract_all_features(sys.argv[1])
        visual = feats["visual"].unsqueeze(0).to(device)
        speech = feats["speech"].unsqueeze(0).to(device)

    # ── Option B: synthetic tensors ──────────
    else:
        print("No video path given — using synthetic tensors.\n")
        B, T_v, T_s = 2, 32, 6
        visual = torch.randn(B, T_v, 1024, device=device)
        speech = torch.randn(B, T_s, 1024, device=device)

    # ── Inference ────────────────────────────
    summaries = model.summarise(visual, speech)
    print("Generated summaries:")
    for i, s in enumerate(summaries):
        print(f"  [{i}] {s[:200]}")

    # ── Training step (mock) ─────────────────
    tokenizer = model.summariser.tokenizer
    dummy_summary = "A person explains machine learning concepts on a whiteboard."
    target = tokenizer(
        dummy_summary,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    ).input_ids.to(device)

    # Repeat target for batch
    if visual.size(0) > 1:
        target = target.expand(visual.size(0), -1)

    model.train()
    out  = model(visual, speech, target_ids=target)
    loss = out["loss"]
    print(f"\nTraining loss (mock): {loss.item():.4f}")

    # ── Parameter summary ─────────────────────
    counts = model.count_parameters()
    print(f"\n{'─'*35}")
    print(f"  Projection   : {counts['projection']:>12,}")
    print(f"  Conformer    : {counts['conformer']:>12,}")
    print(f"  Summariser   : {counts['summariser']:>12,}")
    print(f"{'─'*35}")
    print(f"  Total        : {counts['total']:>12,}")
    print(f"{'─'*35}")
    print("\nFull pipeline complete.")
