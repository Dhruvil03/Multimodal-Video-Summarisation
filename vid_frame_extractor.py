"""
Video Frame Extraction & CLIP Feature Extraction
Phase 1 of Multimodal Video Summarisation Pipeline
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

# pip install opencv-python transformers Pillow
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Frame Sampler
# ─────────────────────────────────────────────

class FrameSampler:
    """
    Samples frames from a video file using one of three strategies:
      - 'uniform'   : evenly spaced across the full duration
      - 'fps'       : every N-th frame at a fixed rate
      - 'keyframe'  : only scene-change keyframes (via OpenCV)
    """

    def __init__(
        self,
        strategy: str = "uniform",
        num_frames: int = 32,
        target_fps: float = 1.0,
        keyframe_threshold: float = 30.0,
    ):
        assert strategy in ("uniform", "fps", "keyframe"), \
            f"Unknown strategy '{strategy}'. Choose: uniform | fps | keyframe"
        self.strategy = strategy
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.keyframe_threshold = keyframe_threshold

    def sample(self, video_path: Union[str, Path]) -> list[np.ndarray]:
        """Returns a list of BGR numpy arrays (H, W, 3)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if self.strategy == "uniform":
            indices = self._uniform_indices(total_frames)
        elif self.strategy == "fps":
            indices = self._fps_indices(total_frames, native_fps)
        else:
            indices = self._keyframe_indices(cap, total_frames)

        frames = self._read_frames(cap, indices)
        cap.release()
        logger.info(f"Sampled {len(frames)} frames from {Path(video_path).name}")
        return frames

    # ── private helpers ──────────────────────

    def _uniform_indices(self, total: int) -> list[int]:
        return np.linspace(0, total - 1, self.num_frames, dtype=int).tolist()

    def _fps_indices(self, total: int, native_fps: float) -> list[int]:
        step = max(1, int(native_fps / self.target_fps))
        return list(range(0, total, step))

    def _keyframe_indices(self, cap: cv2.VideoCapture, total: int) -> list[int]:
        indices, prev_frame = [], None
        for i in range(total):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is None:
                indices.append(i)
            else:
                diff = cv2.absdiff(gray, prev_frame).mean()
                if diff > self.keyframe_threshold:
                    indices.append(i)
            prev_frame = gray
        return indices

    def _read_frames(self, cap: cv2.VideoCapture, indices: list[int]) -> list[np.ndarray]:
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        return frames


# ─────────────────────────────────────────────
# 2. CLIP Feature Extractor
# ─────────────────────────────────────────────

class CLIPVideoEncoder(nn.Module):
    """
    Wraps a frozen HuggingFace CLIP vision encoder.
    Input  : list of PIL images  (or BGR numpy arrays)
    Output : Tensor [T, D]  where D = 768 (ViT-L/14) or 512 (ViT-B/32)
    """

    SUPPORTED_MODELS = {
        "vit-b32":  "openai/clip-vit-base-patch32",
        "vit-b16":  "openai/clip-vit-base-patch16",
        "vit-l14":  "openai/clip-vit-large-patch14",  # recommended
    }

    def __init__(
        self,
        model_name: str = "vit-l14",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        hf_name = self.SUPPORTED_MODELS.get(model_name, model_name)
        logger.info(f"Loading CLIP model: {hf_name}")

        self.processor = CLIPProcessor.from_pretrained(hf_name)
        self.model = CLIPModel.from_pretrained(hf_name).to(self.device)

        # Freeze — we use CLIP purely as a feature extractor
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Feature dimension (e.g. 768 for ViT-L/14)
        self.feature_dim = self.model.config.vision_config.hidden_size
        logger.info(f"CLIP feature dim: {self.feature_dim}, device: {self.device}")

    @torch.no_grad()
    def forward(self, frames: list) -> torch.Tensor:
        """
        Args:
            frames: list of PIL.Image or BGR np.ndarray
        Returns:
            features: Tensor [T, D]
        """
        pil_frames = [self._to_pil(f) for f in frames]
        all_features = []

        for i in range(0, len(pil_frames), self.batch_size):
            batch = pil_frames[i : i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            # vision_model outputs: last_hidden_state + pooler_output
            # pooler_output = CLS token after projection → global frame embedding
            outputs = self.model.vision_model(**inputs)
            pooled = outputs.pooler_output          # [B, D]
            # Optional: L2-normalise (matches CLIP's original contrastive space)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            all_features.append(pooled.cpu())

        return torch.cat(all_features, dim=0)       # [T, D]

    # ── helpers ─────────────────────────────

    @staticmethod
    def _to_pil(frame) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame
        # Assume BGR numpy array from OpenCV
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


# ─────────────────────────────────────────────
# 3. Video Dataset  (for DataLoader use)
# ─────────────────────────────────────────────

class VideoFrameDataset(Dataset):
    """
    Wraps a folder of video files.
    Returns (features, video_id) for each video.
    Useful for pre-computing and caching features.
    """

    def __init__(
        self,
        video_dir: Union[str, Path],
        sampler: FrameSampler,
        encoder: CLIPVideoEncoder,
        extensions: tuple = (".mp4", ".avi", ".mov", ".mkv"),
    ):
        self.paths   = [
            p for p in Path(video_dir).rglob("*") if p.suffix.lower() in extensions
        ]
        self.sampler = sampler
        self.encoder = encoder
        logger.info(f"Found {len(self.paths)} videos in {video_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path    = self.paths[idx]
        frames  = self.sampler.sample(path)
        features = self.encoder(frames)          # [T, D]
        return features, path.stem


# ─────────────────────────────────────────────
# 4. Feature Cache  (save / load .pt)
# ─────────────────────────────────────────────

class FeatureCache:
    """Saves and loads pre-computed CLIP features to avoid re-encoding."""

    def __init__(self, cache_dir: Union[str, Path]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, video_id: str, features: torch.Tensor):
        torch.save(features, self.cache_dir / f"{video_id}.pt")

    def load(self, video_id: str) -> Optional[torch.Tensor]:
        path = self.cache_dir / f"{video_id}.pt"
        return torch.load(path) if path.exists() else None

    def exists(self, video_id: str) -> bool:
        return (self.cache_dir / f"{video_id}.pt").exists()


# ─────────────────────────────────────────────
# 5. End-to-end helper
# ─────────────────────────────────────────────

def extract_clip_features(
    video_path: Union[str, Path],
    strategy: str = "uniform",
    num_frames: int = 32,
    model_name: str = "vit-l14",
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    One-shot convenience function.

    Returns:
        features: Tensor [T, D]  (e.g. [32, 768])
    """
    video_id = Path(video_path).stem

    # Check cache first
    if cache_dir:
        cache = FeatureCache(cache_dir)
        cached = cache.load(video_id)
        if cached is not None:
            logger.info(f"Loaded cached features for '{video_id}'")
            return cached

    sampler  = FrameSampler(strategy=strategy, num_frames=num_frames)
    encoder  = CLIPVideoEncoder(model_name=model_name)

    frames   = sampler.sample(video_path)
    features = encoder(frames)                  # [T, D]

    if cache_dir:
        cache.save(video_id, features)
        logger.info(f"Cached features → {cache_dir}/{video_id}.pt")

    return features


# ─────────────────────────────────────────────
# Quick smoke-test  (runs if called directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else "sample.mp4"

    print("\n=== Multimodal Video Summarisation — Phase 1 ===")
    print(f"Video : {video}")

    features = extract_clip_features(
        video_path=video,
        strategy="uniform",
        num_frames=32,
        model_name="vit-l14",
        cache_dir=".feature_cache",
    )

    print(f"Output shape : {features.shape}")        # e.g. [32, 768]
    print(f"dtype        : {features.dtype}")
    print(f"Sample norms : {features.norm(dim=-1)[:4].tolist()}")
