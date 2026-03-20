"""
Phase 5 — Training Pipeline
Multimodal Video Summarisation on YouCook2

Dataset  : YouCook2 via HuggingFace (lmms-lab/YouCook2)
           3179 val samples split 80/20 into train/val
           Videos downloaded on demand via yt-dlp
GPU      : 16GB  — uses gradient checkpointing + mixed precision
Priority : Balanced (reasonable speed + good performance)

Setup:
    pip install transformers torch torchaudio opencv-python datasets yt-dlp wandb

Run:
    python training_pipeline.py --hf_dir ./youcook2/hf_dataset --output_dir ./checkpoints
"""

import os
import json
import argparse
import logging
import subprocess
import tempfile
import torch.multiprocessing as mp
from pathlib import Path
from typing import Optional

# Must be called before any CUDA initialisation — fixes
# "Cannot re-initialize CUDA in forked subprocess"
mp.set_start_method("spawn", force=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from datasets import load_from_disk

# Local modules
from vid_frame_extractor import extract_clip_features
from speech_feature_extractor import extract_speech_features
from summarisation_head import MultimodalVideoSummariser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. YouCook2 Dataset  (HuggingFace version)
# ─────────────────────────────────────────────

class YouCook2Dataset(Dataset):
    """
    YouCook2 dataset loaded from the HuggingFace disk cache.

    The HF version only has 'val' and 'test' splits — no 'train'.
    We use the val split (3179 samples) and divide it 80/20 manually.

    HF fields per sample:
        id           : "<youtube_id>_<segment_idx>"   e.g. "xHr8X2Wpmno_0"
        youtube_id   : YouTube video ID
        video_url    : full YouTube URL
        recipe_type  : cuisine category id (str)
        segment      : [start_sec, end_sec]
        sentence     : annotation text  ← this is our summary target
        video_path   : relative path (not downloaded — we fetch on demand)

    Videos are downloaded with yt-dlp on first access and cached locally.
    Features are extracted and cached as .pt files.
    """

    def __init__(
        self,
        hf_dir: str,
        split: str = "train",              # "train" or "val" (manual 80/20 split)
        video_dir: str = "./youcook2/videos",
        cache_dir: str = ".feature_cache",
        num_visual_frames: int = 32,
        speech_segment_duration: float = 2.0,
        max_summary_len: int = 64,
        tokenizer=None,
        train_ratio: float = 0.8,
    ):
        self.video_dir  = Path(video_dir)
        self.cache_dir  = Path(cache_dir)
        self.num_visual_frames = num_visual_frames
        self.speech_segment_duration = speech_segment_duration
        self.max_summary_len = max_summary_len
        self.tokenizer  = tokenizer

        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load HF dataset — always from 'val' split (only available)
        hf_ds = load_from_disk(hf_dir)["val"]

        # Manual 80/20 split (fixed seed for reproducibility)
        n_total = len(hf_ds)
        n_train = int(n_total * train_ratio)
        indices = list(range(n_total))

        if split == "train":
            self.samples = [hf_ds[i] for i in indices[:n_train]]
        else:
            self.samples = [hf_ds[i] for i in indices[n_train:]]

        logger.info(
            f"YouCook2 HF {split}: {len(self.samples)}/{n_total} segments "
            f"({'80%' if split == 'train' else '20%'} split)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample   = self.samples[idx]
        yt_id    = sample["youtube_id"]
        seg_id   = sample["id"]
        sentence = sample["sentence"]

        try:
            video_path = self._ensure_video(yt_id)
            visual     = self._load_or_extract_visual(seg_id, video_path)
            speech     = self._load_or_extract_speech(seg_id, video_path)
        except Exception as e:
            logger.warning(f"Skipping {yt_id} ({seg_id}): {e}")
            # Return zero features so the batch can still be formed
            visual = torch.zeros(self.num_visual_frames, 1024)
            speech = torch.zeros(1, 1024)

        target_ids = self._tokenise(sentence)

        return {
            "visual":     visual,
            "speech":     speech,
            "target_ids": target_ids,
            "summary":    sentence,
            "video_id":   yt_id,
        }

    # ── Video download ────────────────────────

    def _ensure_video(self, youtube_id: str) -> str:
        """Download video via yt-dlp forcing H.264 to avoid AV1 decode failures."""
        video_path = self.video_dir / f"{youtube_id}.mp4"
        if video_path.exists():
            return str(video_path)

        url = f"https://www.youtube.com/watch?v={youtube_id}"
        logger.info(f"Downloading {youtube_id}...")
        result = subprocess.run([
            "yt-dlp",
            "-f", "bestvideo[vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]/mp4",
            "--merge-output-format", "mp4",
            "-o", str(video_path),
            "--quiet",
            "--no-warnings",
            url,
        ], capture_output=True, text=True)

        if not video_path.exists():
            raise RuntimeError(
                f"Failed to download {youtube_id}: {result.stderr[-300:]}"
            )
        return str(video_path)

    # ── Feature extraction ────────────────────

    def _load_or_extract_visual(self, seg_id: str, video_path: str) -> torch.Tensor:
        cache_path = self.cache_dir / f"{seg_id}_visual.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)

        features = extract_clip_features(
            video_path,
            strategy="uniform",
            num_frames=self.num_visual_frames,
            cache_dir=None,
        )
        torch.save(features, cache_path)
        return features

    def _load_or_extract_speech(self, seg_id: str, video_path: str) -> torch.Tensor:
        cache_path = self.cache_dir / f"{seg_id}_speech.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)

        features = extract_speech_features(
            video_path,
            model_name="large",
            segment_duration=self.speech_segment_duration,
            cache_dir=None,
        )
        torch.save(features, cache_path)
        return features

    def _tokenise(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(
            text,
            max_length=self.max_summary_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc.input_ids.squeeze(0)   # [L]


# ─────────────────────────────────────────────
# 2. Collate Function  (variable-length padding)
# ─────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """
    Pads visual and speech sequences to the longest in the batch.
    This handles variable-length segments within YouCook2.
    """
    max_v = max(s["visual"].size(0) for s in batch)
    max_s = max(s["speech"].size(0) for s in batch)

    visual_pad, speech_pad, target_pad = [], [], []
    visual_mask, speech_mask = [], []

    for s in batch:
        v, sp = s["visual"], s["speech"]

        # Pad visual
        pad_v = max_v - v.size(0)
        visual_pad.append(F.pad(v, (0, 0, 0, pad_v)))
        visual_mask.append(
            torch.cat([torch.ones(v.size(0)), torch.zeros(pad_v)]).bool()
        )

        # Pad speech
        pad_s = max_s - sp.size(0)
        speech_pad.append(F.pad(sp, (0, 0, 0, pad_s)))
        speech_mask.append(
            torch.cat([torch.ones(sp.size(0)), torch.zeros(pad_s)]).bool()
        )

        target_pad.append(s["target_ids"])

    return {
        "visual":       torch.stack(visual_pad),    # [B, T_v, 1024]
        "speech":       torch.stack(speech_pad),    # [B, T_s, 1024]
        "visual_mask":  torch.stack(visual_mask),   # [B, T_v]
        "speech_mask":  torch.stack(speech_mask),   # [B, T_s]
        "target_ids":   torch.stack(target_pad),    # [B, L]
        "summaries":    [s["summary"] for s in batch],
    }


# Need F for collate_fn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 3. Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    Training loop with:
        - Mixed precision (fp16)     — halves VRAM usage
        - Gradient checkpointing     — trades compute for memory
        - Cosine LR schedule + warmup
        - Gradient clipping          — prevents exploding gradients
        - Checkpoint saving          — best val loss + every N epochs
        - Optional W&B logging
    """

    def __init__(self, args):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on: {self.device}")

        # Build model
        self.model = MultimodalVideoSummariser(
            visual_dim=1024,
            speech_dim=1024,
            d_model=512,
            num_conformer_layers=4,
            num_attn_heads=8,
            gpt2_variant="gpt2",
            max_summary_len=args.max_summary_len,
            dropout=args.dropout,
        ).to(self.device)

        # Gradient checkpointing — trades speed for ~30% VRAM saving
        self.model.conformer.layers.apply(
            lambda m: m.register_forward_hook(
                lambda mod, inp, out: out
            ) if isinstance(m, nn.Linear) else None
        )

        # Tokeniser (from summarisation head — already loaded)
        self.tokenizer = self.model.summariser.tokenizer

        # Datasets
        train_ds = YouCook2Dataset(
            hf_dir=args.hf_dir,
            split="train",
            video_dir=args.video_dir,
            cache_dir=args.cache_dir,
            num_visual_frames=args.num_frames,
            speech_segment_duration=args.speech_seg_dur,
            max_summary_len=args.max_summary_len,
            tokenizer=self.tokenizer,
        )
        val_ds = YouCook2Dataset(
            hf_dir=args.hf_dir,
            split="val",
            video_dir=args.video_dir,
            cache_dir=args.cache_dir,
            num_visual_frames=args.num_frames,
            speech_segment_duration=args.speech_seg_dur,
            max_summary_len=args.max_summary_len,
            tokenizer=self.tokenizer,
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,        # 0 = no forking, avoids CUDA reinit error
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Optimiser — separate LRs for pre-trained vs new params
        pretrained_params = list(self.model.summariser.gpt2.parameters())
        new_params = (
            list(self.model.projection.parameters()) +
            list(self.model.conformer.parameters()) +
            list(self.model.summariser.cross_attn_layers.parameters()) +
            list(self.model.summariser.enc_proj.parameters())
        )
        self.optimizer = torch.optim.AdamW([
            {"params": pretrained_params, "lr": args.lr * 0.1},  # lower LR for GPT-2
            {"params": new_params,        "lr": args.lr},
        ], weight_decay=args.weight_decay)

        # LR scheduler — cosine with warmup
        total_steps   = len(self.train_loader) * args.epochs
        warmup_steps  = total_steps // 10
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision scaler
        self.scaler = GradScaler("cuda")

        # State
        self.best_val_loss = float("inf")
        self.output_dir    = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # W&B (optional)
        self.use_wandb = args.wandb
        if self.use_wandb:
            import wandb
            wandb.init(project="multimodal-video-summ", config=vars(args))

        self._log_param_count()

    def _log_param_count(self):
        counts = self.model.count_parameters()
        logger.info("Parameter counts:")
        for k, v in counts.items():
            logger.info(f"  {k:<15}: {v:>12,}")

    # ── Training epoch ────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for step, batch in enumerate(self.train_loader):
            visual     = batch["visual"].to(self.device)
            speech     = batch["speech"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            self.optimizer.zero_grad()

            with autocast("cuda"):
                out  = self.model(visual, speech, target_ids=target_ids)
                loss = out["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

            if step % args.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch:02d} | Step {step:04d}/{len(self.train_loader)} "
                    f"| Loss {loss.item():.4f} | LR {lr:.2e}"
                )
                if self.use_wandb:
                    import wandb
                    wandb.log({"train/loss": loss.item(), "train/lr": lr})

        return total_loss / n_batches

    # ── Validation epoch ─────────────────────

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0

        for batch in self.val_loader:
            visual     = batch["visual"].to(self.device)
            speech     = batch["speech"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            with autocast("cuda"):
                out  = self.model(visual, speech, target_ids=target_ids)
                loss = out["loss"]

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches
        logger.info(f"Epoch {epoch:02d} | Val loss: {avg_loss:.4f}")

        if self.use_wandb:
            import wandb
            wandb.log({"val/loss": avg_loss, "epoch": epoch})

        return avg_loss

    # ── Sample generation (qualitative check) ─

    @torch.no_grad()
    def generate_samples(self, epoch: int, n: int = 3):
        self.model.eval()
        batch = next(iter(self.val_loader))
        visual = batch["visual"][:n].to(self.device)
        speech = batch["speech"][:n].to(self.device)

        summaries = self.model.summarise(visual, speech)

        logger.info(f"\n── Sample summaries (epoch {epoch}) ──")
        for i, (gen, ref) in enumerate(zip(summaries, batch["summaries"][:n])):
            logger.info(f"  REF : {ref}")
            logger.info(f"  GEN : {gen}")
            logger.info("")

    # ── Checkpoint save/load ──────────────────

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        state = {
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "scaler":     self.scaler.state_dict(),
        }
        # Always save latest
        torch.save(state, self.output_dir / "latest.pt")

        # Save best
        if is_best:
            torch.save(state, self.output_dir / "best.pt")
            logger.info(f"  New best model saved (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % self.args.save_every == 0:
            torch.save(state, self.output_dir / f"epoch_{epoch:03d}.pt")

    def load_checkpoint(self, path: str):
        if not Path(path).exists():
            logger.info(f"No checkpoint found at {path} — starting from scratch")
            return 0
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        logger.info(f"Resumed from {path} (epoch {state['epoch']})")
        return state["epoch"]

    # ── Main training loop ───────────────────

    def train(self):
        start_epoch = 0

        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume)

        logger.info(f"\nStarting training for {self.args.epochs} epochs\n")

        for epoch in range(start_epoch + 1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.val_epoch(epoch)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)
            self.generate_samples(epoch)

            logger.info(
                f"Epoch {epoch:02d} complete | "
                f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"Best {self.best_val_loss:.4f}"
            )

        logger.info("Training complete.")
        if self.use_wandb:
            import wandb
            wandb.finish()


# ─────────────────────────────────────────────
# 4. Argument parser
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Train Multimodal Video Summariser")

    # Paths
    p.add_argument("--hf_dir",      default="./youcook2/hf_dataset", help="HuggingFace dataset dir")
    p.add_argument("--video_dir",   default="./youcook2/videos",     help="Downloaded videos dir")
    p.add_argument("--output_dir",  default="./checkpoints",         help="Save checkpoints here")
    p.add_argument("--cache_dir",   default=".feature_cache",        help="Pre-computed features")
    p.add_argument("--resume",      default=None,                    help="Resume from checkpoint path")

    # Data
    p.add_argument("--num_frames",      type=int,   default=32)
    p.add_argument("--speech_seg_dur",  type=float, default=2.0)
    p.add_argument("--max_summary_len", type=int,   default=64)

    # Training
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=4)    # 16GB safe default
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--num_workers",  type=int,   default=4)

    # Logging
    p.add_argument("--log_every",  type=int,  default=50)
    p.add_argument("--save_every", type=int,  default=5)
    p.add_argument("--wandb",      action="store_true", help="Enable W&B logging")

    return p.parse_args()


# ─────────────────────────────────────────────
# 5. SLURM job script (printed to stdout)
# ─────────────────────────────────────────────

SLURM_SCRIPT = """#!/bin/bash
#SBATCH --job-name=vid-summ-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

mkdir -p logs
conda activate video-summ
cd ~/mul_Vid_Summ

python training_pipeline.py \\
    --hf_dir      ./youcook2/hf_dataset \\
    --video_dir   ./youcook2/videos \\
    --output_dir  ./checkpoints \\
    --cache_dir   .feature_cache \\
    --epochs      20 \\
    --batch_size  4 \\
    --lr          3e-4 \\
    --log_every   50 \\
    --wandb
"""


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()

    # Print SLURM script if requested
    if "--print_slurm" in __import__("sys").argv:
        print(SLURM_SCRIPT)
        exit(0)

    trainer = Trainer(args)
    trainer.train()
