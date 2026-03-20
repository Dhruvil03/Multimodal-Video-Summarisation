"""
Speech Feature Extraction — Phase 1b
Multimodal Video Summarisation Pipeline

Extracts audio from a video file and produces frame-level
acoustic embeddings using a frozen Wav2Vec2 encoder.

Requires:
    pip install transformers torchaudio ffmpeg-python
    sudo apt-get install ffmpeg   (or: conda install -c conda-forge ffmpeg)

Output shape: [T_s, 1024]  (Wav2Vec2-large)
Plugs into:   ProjectionAlignmentModule(speech_features=...)
"""

import torch
import torch.nn as nn
import torchaudio
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, Union

from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000   # Wav2Vec2 expects 16 kHz mono


# ─────────────────────────────────────────────
# 1. Audio Extractor  (video → waveform)
# ─────────────────────────────────────────────

class AudioExtractor:
    """
    Extracts the audio track from a video file using FFmpeg,
    resamples to 16 kHz mono, and returns a raw waveform tensor.

    Why FFmpeg?  torchaudio.load() handles .wav/.mp3 natively but
    struggles with embedded audio in .mp4/.mkv containers. FFmpeg
    is the most reliable cross-format solution on cluster environments.
    """

    def __init__(self, target_sr: int = TARGET_SAMPLE_RATE):
        self.target_sr = target_sr
        self._check_ffmpeg()

    def extract(self, video_path: Union[str, Path]) -> torch.Tensor:
        """
        Args:
            video_path: path to any video file (.mp4, .mkv, .avi, ...)

        Returns:
            waveform: [1, N]  float32 tensor at target_sr
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp_path = tmp.name

        # FFmpeg: extract audio, resample, convert to mono WAV
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",                          # no video
            "-acodec", "pcm_s16le",         # 16-bit PCM
            "-ar", str(self.target_sr),     # resample to 16 kHz
            "-ac", "1",                     # mono
            tmp_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed for {video_path.name}:\n{result.stderr[-500:]}"
            )

        waveform, sr = torchaudio.load(tmp_path)   # [1, N]
        Path(tmp_path).unlink(missing_ok=True)

        # Sanity-check sample rate
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        logger.info(
            f"Audio extracted: {waveform.shape[1] / self.target_sr:.1f}s "
            f"@ {self.target_sr} Hz from {video_path.name}"
        )
        return waveform                             # [1, N]

    @staticmethod
    def _check_ffmpeg():
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True
        )
        if result.returncode != 0:
            raise EnvironmentError(
                "FFmpeg not found. Install with:\n"
                "  conda install -c conda-forge ffmpeg\n"
                "  or: sudo apt-get install ffmpeg"
            )


# ─────────────────────────────────────────────
# 2. Wav2Vec2 Speech Encoder
# ─────────────────────────────────────────────

class Wav2Vec2SpeechEncoder(nn.Module):
    """
    Frozen Wav2Vec2-large encoder that produces frame-level hidden states.

    Wav2Vec2 processes raw waveform → CNN feature encoder → Transformer.
    We tap the final Transformer hidden states, giving one 1024-d vector
    per ~20ms of audio (50 frames/sec at 16 kHz).

    For a 60-second video this gives ~3000 speech tokens — which is a lot.
    Use segment_duration to chunk into fixed-size windows and pool within
    each chunk, reducing T_s to something manageable (e.g. 32–128 tokens).
    """

    SUPPORTED_MODELS = {
        "base":  "facebook/wav2vec2-base",           # 768-d, faster
        "large": "facebook/wav2vec2-large",           # 1024-d, recommended
        "large-lv60": "facebook/wav2vec2-large-lv60", # best quality
    }

    def __init__(
        self,
        model_name: str = "large",
        device: Optional[str] = None,
        segment_duration: float = 2.0,   # seconds per output token
        batch_size: int = 8,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.segment_duration = segment_duration
        self.batch_size = batch_size

        hf_name = self.SUPPORTED_MODELS.get(model_name, model_name)
        logger.info(f"Loading Wav2Vec2: {hf_name}")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hf_name)
        self.model = Wav2Vec2Model.from_pretrained(hf_name).to(self.device)

        # Freeze all weights — used as a fixed feature extractor
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.feature_dim = self.model.config.hidden_size   # 1024 for large
        logger.info(
            f"Wav2Vec2 feature dim: {self.feature_dim}, "
            f"segment: {segment_duration}s, device: {self.device}"
        )

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [1, N]  raw audio at 16 kHz

        Returns:
            features: [T_s, 1024]
            where T_s = number of segments (ceil(duration / segment_duration))
        """
        waveform = waveform.squeeze(0)           # [N]
        seg_len  = int(self.segment_duration * TARGET_SAMPLE_RATE)
        segments = waveform.split(seg_len)       # tuple of [seg_len] tensors

        all_features = []

        for i in range(0, len(segments), self.batch_size):
            batch_segs = segments[i : i + self.batch_size]

            # Pad all segments in the batch to the same length
            max_len = max(s.size(0) for s in batch_segs)
            padded  = torch.stack([
                torch.nn.functional.pad(s, (0, max_len - s.size(0)))
                for s in batch_segs
            ])                                   # [B, max_len]

            # Wav2Vec2FeatureExtractor expects numpy or list of arrays
            inputs = self.feature_extractor(
                padded.numpy(),
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            hidden  = outputs.last_hidden_state  # [B, T_frames, 1024]

            # Mean-pool each segment's frame-level states → one vector/segment
            pooled  = hidden.mean(dim=1)         # [B, 1024]
            all_features.append(pooled.cpu())

        features = torch.cat(all_features, dim=0)   # [T_s, 1024]
        logger.info(f"Speech features: {features.shape}")
        return features


# ─────────────────────────────────────────────
# 3. Feature Cache  (reuse from Phase 1 pattern)
# ─────────────────────────────────────────────

class SpeechFeatureCache:
    """Saves/loads pre-computed speech features to avoid re-encoding."""

    def __init__(self, cache_dir: Union[str, Path]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, video_id: str, features: torch.Tensor):
        torch.save(features, self.cache_dir / f"{video_id}_speech.pt")

    def load(self, video_id: str) -> Optional[torch.Tensor]:
        path = self.cache_dir / f"{video_id}_speech.pt"
        return torch.load(path) if path.exists() else None


# ─────────────────────────────────────────────
# 4. End-to-end helper
# ─────────────────────────────────────────────

def extract_speech_features(
    video_path: Union[str, Path],
    model_name: str = "large",
    segment_duration: float = 2.0,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    One-shot convenience function.

    Returns:
        features: Tensor [T_s, 1024]
    """
    video_id = Path(video_path).stem

    if cache_dir:
        cache = SpeechFeatureCache(cache_dir)
        cached = cache.load(video_id)
        if cached is not None:
            logger.info(f"Loaded cached speech features for '{video_id}'")
            return cached

    extractor = AudioExtractor()
    encoder   = Wav2Vec2SpeechEncoder(
        model_name=model_name,
        segment_duration=segment_duration,
    )

    waveform = extractor.extract(video_path)         # [1, N]
    features = encoder(waveform)                     # [T_s, 1024]

    if cache_dir:
        cache = SpeechFeatureCache(cache_dir)
        cache.save(video_id, features)
        logger.info(f"Cached speech features → {cache_dir}/{video_id}_speech.pt")

    return features


# ─────────────────────────────────────────────
# 5. Combined Phase 1 + 1b runner
#    (feeds directly into Phase 2)
# ─────────────────────────────────────────────

def extract_all_features(
    video_path: Union[str, Path],
    num_visual_frames: int = 32,
    speech_segment_duration: float = 2.0,
    cache_dir: str = ".feature_cache",
) -> dict:
    """
    Runs both Phase 1 (CLIP) and Phase 1b (Wav2Vec2) and returns
    a dict ready to pass into ProjectionAlignmentModule.

    Returns:
        {
          "visual":  Tensor [T_v, 768],
          "speech":  Tensor [T_s, 1024],
          "video_id": str,
        }
    """
    # Import Phase 1 extractor
    from vid_frame_extractor import extract_clip_features

    video_id = Path(video_path).stem
    logger.info(f"\n{'='*50}")
    logger.info(f"Extracting all features for: {video_id}")
    logger.info(f"{'='*50}")

    visual  = extract_clip_features(
        video_path,
        strategy="uniform",
        num_frames=num_visual_frames,
        cache_dir=cache_dir,
    )

    speech  = extract_speech_features(
        video_path,
        model_name="large",
        segment_duration=speech_segment_duration,
        cache_dir=cache_dir,
    )

    logger.info(f"\nVisual features : {visual.shape}")   # [T_v, 768]
    logger.info(f"Speech features : {speech.shape}")    # [T_s, 1024]
    logger.info(f"{'='*50}\n")

    return {
        "visual":   visual,
        "speech":   speech,
        "video_id": video_id,
    }


# ─────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else "sample.mp4"

    print("\n=== Phase 1b — Speech Feature Extraction ===")

    features = extract_all_features(
        video_path=video,
        num_visual_frames=32,
        speech_segment_duration=2.0,
        cache_dir=".feature_cache",
    )

    visual = features["visual"]
    speech = features["speech"]

    print(f"\nVisual : {visual.shape}")    # e.g. [32, 768]
    print(f"Speech : {speech.shape}")     # e.g. [30, 1024]  for 60s video

    # ── Feed straight into Phase 2 ────────────
    from projection_alignment import build_projection_module

    module = build_projection_module(d_model=512)

    visual_b = visual.unsqueeze(0)         # [1, T_v, 768]
    speech_b = speech.unsqueeze(0)         # [1, T_s, 1024]

    fused, mask = module(visual_b, speech_b)
    print(f"\nPhase 2 fused   : {fused.shape}")   # [1, T_v+T_s, 512]
    print(f"Padding mask    : {mask.shape}")
    print("\nPhase 1 + 1b + 2 pipeline complete — ready for Conformer.")
