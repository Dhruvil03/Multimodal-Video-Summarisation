# Multimodal Video Summarisation

> An end-to-end single-sequence model that watches a video, listens to its audio, fuses both over time using a Conformer encoder, and writes a text summary of what happened.

**Stack:** PyTorch · CLIP ViT-L/14 · Wav2Vec2 · Conformer · GPT-2 · YouCook2

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Dataset Setup](#dataset-setup-youcook2)
- [Training](#training)
- [Key Design Decisions](#key-design-decisions)
- [Troubleshooting](#troubleshooting)
- [Cluster Quick Reference](#cluster-quick-reference)

---

## Overview

Most video summarisation systems handle vision or speech separately. This project builds a **single unified model** that reads both streams together and generates a concise text summary.

Rather than summarising vision and speech independently and merging the results, the Conformer encoder sees both modalities as **one unified token sequence** — learning cross-modal correlations like a speaker saying *"as you can see here"* aligning with what is on screen at that moment.

---

## Architecture

### Pipeline Flow

```
Video file (.mp4)
   │
   ├── Phase 1:  CLIP ViT-L/14  ──────────►  Visual features  [T_v, 1024]
   │                                                   │
   └── Phase 1b: Wav2Vec2-large ──────────►  Speech features  [T_s, 1024]
                                                        │
                             Phase 2: Projection ───────┤
                             Linear(1024→512) + LayerNorm + Temporal PE
                                                        │
                                             [B, T_v+T_s, 512]
                                                        │
                             Phase 3: Conformer Encoder (4 blocks)
                             MHSA + Depthwise Conv + FFN × 4
                                                        │
                                             [B, T_v+T_s, 512]
                                                        │
                             Phase 4: GPT-2 Decoder
                             Cross-attention over encoder output
                                                        │
                                          Summary text (generated)
```

### Phase Summary

| Phase | File | Input | Output |
|-------|------|-------|--------|
| 1 — Visual extraction | `vid_frame_extractor.py` | Video `.mp4` | `[T_v, 1024]` |
| 1b — Speech extraction | `speech_feature_extractor.py` | Video `.mp4` | `[T_s, 1024]` |
| 2 — Projection & alignment | `projection_alignment.py` | `[T_v/T_s, 1024]` | `[B, T, 512]` |
| 3 — Conformer encoder | `conformer_encoder.py` | `[B, T, 512]` | `[B, T, 512]` |
| 4 — Summarisation head | `summarisation_head.py` | `[B, T, 512]` | Summary text |
| 5 — Training pipeline | `training_pipeline.py` | YouCook2 dataset | Trained model |

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Projection module | 1,052,672 |
| Conformer encoder | 24,242,176 |
| Summarisation head (GPT-2) | 181,550,592 |
| **Total** | **206,845,440** |

---

## File Structure

```
mul_Vid_Summ/
├── vid_frame_extractor.py       Phase 1  — CLIP visual feature extraction
├── speech_feature_extractor.py  Phase 1b — Wav2Vec2 speech feature extraction
├── projection_alignment.py      Phase 2  — Project & fuse to [B, T, 512]
├── conformer_encoder.py         Phase 3  — Conformer temporal encoder
├── summarisation_head.py        Phase 4  — Full model + GPT-2 decoder
├── training_pipeline.py         Phase 5  — Training loop (YouCook2)
├── youcook2/
│   ├── hf_dataset/              HuggingFace YouCook2 dataset
│   └── videos/                  Downloaded YouTube videos
├── checkpoints/
│   ├── latest.pt                Most recent epoch checkpoint
│   └── best.pt                  Best validation loss checkpoint
└── .feature_cache/              Cached CLIP + Wav2Vec2 .pt files
```

---

## Installation

### 1. Create environment

```bash
conda create -n video-summ python=3.10 -y
conda activate video-summ
```

### 2. Install dependencies

```bash
# Match cu118/cu121 to your CUDA version (check with: nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers torchaudio opencv-python Pillow
pip install datasets yt-dlp wandb
conda install -c conda-forge ffmpeg
```

---

## Running the Pipeline

### Phase 1 — Visual Feature Extraction

Extracts CLIP ViT-L/14 features from uniformly sampled video frames.

```bash
python vid_frame_extractor.py sample/sample.mp4
# Output: torch.Size([32, 1024])
# Cached to: .feature_cache/sample.pt
```

### Phase 1b — Speech Feature Extraction

Extracts audio via FFmpeg, runs Wav2Vec2-large, and mean-pools per 2-second segment.

```bash
python speech_feature_extractor.py sample/sample.mp4
# Visual : torch.Size([32, 1024])
# Speech : torch.Size([6, 1024])
# Phase 2 fused: torch.Size([1, 38, 512])
```

### Phase 3 — Conformer Encoder Test

Runs Phases 1 + 1b + 2 + 3 end-to-end.

```bash
python conformer_encoder.py sample/sample.mp4
# Phase 2 output : torch.Size([1, 38, 512])
# Phase 3 output : torch.Size([1, 38, 512])
```

### Phase 4 — Full Model Inference

Runs the complete pipeline and generates a summary.

```bash
python summarisation_head.py sample/sample.mp4
# Generated summaries:
#   [0] <generated text>
# Total params: 206,845,440
```

---

## Dataset Setup (YouCook2)

### Download via HuggingFace

```bash
pip install datasets

python -c "
from datasets import load_dataset
ds = load_dataset('lmms-lab/YouCook2')
ds.save_to_disk('./youcook2/hf_dataset')
print(ds)
"
```

The HuggingFace version contains:
- **3,179** annotated segments in the `val` split
- **1,471** segments in the `test` split
- Fields: `id`, `youtube_id`, `video_url`, `recipe_type`, `segment`, `sentence`

> **Note:** The HF dataset has no `train` split. The pipeline automatically divides `val` 80/20 → **2,543 train / 636 val** segments.

---

## Training

### Start training

```bash
python training_pipeline.py \
    --hf_dir     ./youcook2/hf_dataset \
    --video_dir  ./youcook2/videos \
    --output_dir ./checkpoints \
    --epochs     7 \
    --batch_size 4 \
    --resume     ./checkpoints/latest.pt
```

> **Resume behaviour:** If `latest.pt` does not exist, training starts from scratch automatically. If it exists, training resumes from the last completed epoch. Use the same command every time.

### SLURM job (university cluster)

```bash
python training_pipeline.py --print_slurm > run_training.sh
sbatch run_training.sh

# Monitor:
squeue -u <your_username>
tail -f logs/train_<job_id>.out
```

### Expected learning curve

| Epoch | Train Loss | Quality |
|-------|-----------|---------|
| 1 | ~5.5 | Random text |
| 3 | ~4.0 | Cooking words appear |
| 5 | ~3.2 | Rough but on-topic |
| 7 | ~2.8 | Recognisable summaries |
| 10 | ~2.5 | Good summaries |

### Time estimates (16GB GPU)

| Run | Time |
|-----|------|
| First epoch (download + feature extraction) | 5–9 hours |
| Each subsequent epoch (cached features) | 15–30 mins |
| 7 epochs total (after first) | ~6–10 hours |

---

## Key Design Decisions

### Why Conformer?
A standard Transformer captures global context via self-attention but misses local temporal patterns. A CNN captures local patterns but lacks long-range context. The Conformer combines both in every layer — **self-attention for global context** (what is this video about?) and **depthwise convolution with kernel=31 for local patterns** (what is happening in this 2-second window?).

### Why freeze CLIP and Wav2Vec2?
Both models were pre-trained on massive datasets (400M image-text pairs for CLIP, 60k hours of audio for Wav2Vec2). Freezing preserves their rich representations and reduces GPU memory. Only the projection layers, Conformer, and GPT-2 decoder are trained from scratch.

### Why two learning rates?
GPT-2 is pre-trained and receives `lr × 0.1` (3e-5) to preserve its language knowledge. The Conformer, projection layers, and cross-attention layers are trained from scratch at full `lr` (3e-4).

### Feature caching
CLIP and Wav2Vec2 inference is expensive. Features are extracted once and saved as `.pt` files in `.feature_cache/`. The first epoch is slow; every subsequent epoch loads cached features instantly.

### Modality type embeddings
Each token gets a small learned embedding identifying it as visual (0) or speech (1) — analogous to BERT's segment embeddings. This lets the Conformer learn that visual and speech tokens behave differently.

---

## Troubleshooting

### `RuntimeError: Cannot re-initialize CUDA in forked subprocess`
DataLoader workers fork after CUDA is initialised. Fixed by:
```python
mp.set_start_method("spawn", force=True)   # top of training_pipeline.py
# and num_workers=0 in DataLoader
```

### `mat1 and mat2 shapes cannot be multiplied (32x1024 and 768x512)`
CLIP ViT-L/14 `pooler_output` is **1024-d** (not 768). Ensure `build_projection_module` uses `visual_dim=1024`.

### `FutureWarning: torch.cuda.amp.GradScaler is deprecated`
Update to the new API:
```python
# Old
GradScaler()
autocast()

# New
GradScaler("cuda")
autocast("cuda")
```

### `ERROR 404: Not Found` on YouCook2 download
The original `youcook2.eecs.umich.edu` server is down. Use the HuggingFace mirror:
```bash
ds = load_dataset('lmms-lab/YouCook2')
```

### Repetitive generated text before training
Expected — untrained cross-attention produces degenerate output. Summaries improve significantly after 3–5 epochs of fine-tuning.

---

## Cluster Quick Reference

| Task | Command |
|------|---------|
| Connect | `ssh yourname@cluster.university.ac.uk` |
| Load Python | `module load python/3.10` |
| Activate env | `conda activate video-summ` |
| Submit job | `sbatch run_training.sh` |
| Check status | `squeue -u <username>` |
| Watch logs | `tail -f logs/train_<job_id>.out` |
| Cancel job | `scancel <job_id>` |
| Check GPU | `nvidia-smi` |

---

## References

- [CLIP — Radford et al. 2021](https://arxiv.org/abs/2103.00020)
- [Wav2Vec2 — Baevski et al. 2020](https://arxiv.org/abs/2006.11477)
- [Conformer — Gulati et al. 2020](https://arxiv.org/abs/2005.08100)
- [GPT-2 — Radford et al. 2019](https://openai.com/research/language-unsupervised)
- [YouCook2 Dataset](http://youcook2.eecs.umich.edu/)
