"""
Evaluation & Inference Script
Multimodal Video Summarisation Pipeline

Two modes:
  1. evaluate  — compute ROUGE-1/2/L scores on YouCook2 val split
  2. infer     — generate a summary for any new video file

Usage:
    # Evaluate on val split
    python evaluate.py evaluate \
        --checkpoint ./checkpoints/best.pt \
        --hf_dir     ./youcook2/hf_dataset \
        --video_dir  ./youcook2/videos

    # Infer on a new video
    python evaluate.py infer \
        --checkpoint ./checkpoints/best.pt \
        --video      ./my_video.mp4

Install:
    pip install rouge-score
"""

import argparse
import logging
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Model loader
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str) -> object:
    """Load the full model from a checkpoint."""
    from summarisation_head import MultimodalVideoSummariser

    model = MultimodalVideoSummariser(
        visual_dim=1024,
        speech_dim=1024,
        d_model=512,
        num_conformer_layers=4,
        num_attn_heads=8,
        gpt2_variant="gpt2",
        max_summary_len=150,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    epoch    = state.get("epoch", "?")
    val_loss = state.get("val_loss", "?")
    logger.info(f"Loaded checkpoint — epoch {epoch}, val_loss {val_loss:.4f}")
    return model


# ─────────────────────────────────────────────
# 2. Improved generation  (temperature + top-p)
# ─────────────────────────────────────────────

@torch.no_grad()
def generate_summary(
    model,
    visual: torch.Tensor,          # [1, T_v, 1024]
    speech: torch.Tensor,          # [1, T_s, 1024]
    device: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_len: int = 100,
) -> str:
    """
    Generate a summary using temperature + top-p nucleus sampling.
    Much better than greedy decoding — avoids repetitive output.

    temperature: lower = more focused, higher = more creative
    top_p:       nucleus size — sample only from top 90% probability mass
    """
    visual = visual.to(device)
    speech = speech.to(device)

    # Phase 2 + 3: get encoder output
    fused, mask   = model.projection(visual, speech)
    encoded       = model.conformer(fused, src_mask=mask)

    # Project encoder to GPT-2 dim
    enc = model.summariser.enc_proj(encoded)

    tokenizer = model.summariser.tokenizer
    bos_id    = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos_id    = tokenizer.eos_token_id

    generated = torch.full((1, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_len):
        hidden = model.summariser.gpt2.transformer.wte(generated)
        hidden = hidden + model.summariser.gpt2.transformer.wpe(
            torch.arange(hidden.size(1), device=device)
        )
        for i, block in enumerate(model.summariser.gpt2.transformer.h):
            hidden = block(hidden)[0]
            hidden = model.summariser.cross_attn_layers[i](hidden, enc, mask)

        hidden = model.summariser.gpt2.transformer.ln_f(hidden)
        logits = model.summariser.gpt2.lm_head(hidden[:, -1, :])  # [1, vocab]

        # Temperature scaling
        logits = logits / temperature

        # Top-p nucleus filtering
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_idx_to_remove = cumulative_probs > top_p
        sorted_idx_to_remove[:, 1:] = sorted_idx_to_remove[:, :-1].clone()
        sorted_idx_to_remove[:, 0]  = False
        logits[0, sorted_idx[0, sorted_idx_to_remove[0]]] = float("-inf")

        probs      = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == eos_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# 3. ROUGE Evaluation
# ─────────────────────────────────────────────

def evaluate(args):
    """
    Run ROUGE evaluation on the YouCook2 val split.
    Saves per-sample results to a JSON file and prints summary scores.
    """
    from datasets import load_from_disk
    from speech_feature_extractor import extract_all_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(args.checkpoint, device)

    # Load val split (last 20% of HF val set)
    hf_ds   = load_from_disk(args.hf_dir)["val"]
    n_total = len(hf_ds)
    n_train = int(n_total * 0.8)
    val_samples = [hf_ds[i] for i in range(n_train, n_total)]

    if args.num_samples:
        val_samples = val_samples[:args.num_samples]

    logger.info(f"Evaluating on {len(val_samples)} samples...")

    scorer  = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    results = []
    r1_scores, r2_scores, rL_scores = [], [], []

    for i, sample in enumerate(val_samples):
        yt_id    = sample["youtube_id"]
        seg_id   = sample["id"]
        ref_text = sample["sentence"]

        # ── Load or extract features ──────────
        try:
            video_path = str(Path(args.video_dir) / f"{yt_id}.mp4")
            if not Path(video_path).exists():
                logger.warning(f"Video not found, skipping: {yt_id}")
                continue

            vis_cache = Path(args.cache_dir) / f"{seg_id}_visual.pt"
            spe_cache = Path(args.cache_dir) / f"{seg_id}_speech.pt"

            if vis_cache.exists() and spe_cache.exists():
                visual = torch.load(vis_cache, weights_only=True).unsqueeze(0)
                speech = torch.load(spe_cache, weights_only=True).unsqueeze(0)
            else:
                feats  = extract_all_features(video_path, cache_dir=args.cache_dir)
                visual = feats["visual"].unsqueeze(0)
                speech = feats["speech"].unsqueeze(0)

        except Exception as e:
            logger.warning(f"Skipping {yt_id}: {e}")
            continue

        # ── Generate summary ──────────────────
        gen_text = generate_summary(
            model, visual, speech, device,
            temperature=args.temperature,
            top_p=args.top_p,
            max_len=args.max_len,
        )

        # ── ROUGE scores ──────────────────────
        scores = scorer.score(ref_text, gen_text)
        r1 = scores["rouge1"].fmeasure
        r2 = scores["rouge2"].fmeasure
        rL = scores["rougeL"].fmeasure

        r1_scores.append(r1)
        r2_scores.append(r2)
        rL_scores.append(rL)

        results.append({
            "id":       seg_id,
            "ref":      ref_text,
            "gen":      gen_text,
            "rouge1":   round(r1, 4),
            "rouge2":   round(r2, 4),
            "rougeL":   round(rL, 4),
        })

        if (i + 1) % 10 == 0:
            logger.info(
                f"[{i+1}/{len(val_samples)}] "
                f"R1={sum(r1_scores)/len(r1_scores):.4f} "
                f"R2={sum(r2_scores)/len(r2_scores):.4f} "
                f"RL={sum(rL_scores)/len(rL_scores):.4f}"
            )

    # ── Final scores ──────────────────────────
    n = len(r1_scores)
    final = {
        "num_samples": n,
        "rouge1":  round(sum(r1_scores) / n, 4),
        "rouge2":  round(sum(r2_scores) / n, 4),
        "rougeL":  round(sum(rL_scores) / n, 4),
    }

    print("\n" + "="*45)
    print("  EVALUATION RESULTS")
    print("="*45)
    print(f"  Samples evaluated : {n}")
    print(f"  ROUGE-1           : {final['rouge1']:.4f}")
    print(f"  ROUGE-2           : {final['rouge2']:.4f}")
    print(f"  ROUGE-L           : {final['rougeL']:.4f}")
    print("="*45)

    # ── Save results ──────────────────────────
    out_path = Path(args.output_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": final, "samples": results}, f, indent=2)
    logger.info(f"Results saved to {out_path}")


# ─────────────────────────────────────────────
# 4. Inference on a new video
# ─────────────────────────────────────────────

def infer(args):
    """Generate a summary for any video file."""
    from speech_feature_extractor import extract_all_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(args.checkpoint, device)

    logger.info(f"Extracting features from: {args.video}")
    feats  = extract_all_features(
        args.video,
        num_visual_frames=32,
        speech_segment_duration=2.0,
        cache_dir=args.cache_dir,
    )
    visual = feats["visual"].unsqueeze(0)   # [1, T_v, 1024]
    speech = feats["speech"].unsqueeze(0)   # [1, T_s, 1024]

    logger.info("Generating summary...")
    summary = generate_summary(
        model, visual, speech, device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_len=args.max_len,
    )

    print("\n" + "="*45)
    print("  VIDEO SUMMARY")
    print("="*45)
    print(f"  Video   : {Path(args.video).name}")
    print(f"  Summary : {summary}")
    print("="*45)
    return summary


# ─────────────────────────────────────────────
# 5. Argument parser
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Evaluate or run inference")
    sub = p.add_subparsers(dest="mode", required=True)

    # ── Shared args ───────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--checkpoint",   default="./checkpoints/best.pt")
    shared.add_argument("--cache_dir",    default=".feature_cache")
    shared.add_argument("--temperature",  type=float, default=0.7)
    shared.add_argument("--top_p",        type=float, default=0.9)
    shared.add_argument("--max_len",      type=int,   default=100)

    # ── Evaluate mode ─────────────────────────
    ev = sub.add_parser("evaluate", parents=[shared])
    ev.add_argument("--hf_dir",      default="./youcook2/hf_dataset")
    ev.add_argument("--video_dir",   default="./youcook2/videos")
    ev.add_argument("--output_dir",  default="./eval_output")
    ev.add_argument("--num_samples", type=int, default=None,
                    help="Limit number of val samples (default: all)")

    # ── Infer mode ────────────────────────────
    inf = sub.add_parser("infer", parents=[shared])
    inf.add_argument("--video", required=True, help="Path to video file")

    return p.parse_args()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    if args.mode == "evaluate":
        evaluate(args)
    else:
        infer(args)
