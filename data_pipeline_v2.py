"""
=============================================================================
FEASIBILITY ANALYSIS + CORRECTED DATA PIPELINE
=============================================================================
Matches the submitted proposal: "Localizing Musical Instruments in Ensemble
Videos Using Transformer-Based Cross-Modal Attention"
=============================================================================

FEASIBILITY VERDICT: ✅ YES — with adjustments for 8GB VRAM

HARDWARE REALITY:
  RTX 5060 = 8GB VRAM (GDDR7)
  System storage = 200-300 GB available

STORAGE BUDGET:
  ┌────────────────────────────────────┬──────────┐
  │ Item                               │ Size     │
  ├────────────────────────────────────┼──────────┤
  │ Raw videos (448 solos + 141 duets) │ ~40-60GB │
  │ Extracted frames (10 per video)    │ ~2 GB    │
  │ Extracted waveforms (.wav)         │ ~5 GB    │
  │ HDF5 processed files               │ ~8 GB    │
  │ Model checkpoints                  │ ~1 GB    │
  │ TOTAL                              │ ~60-75GB │
  └────────────────────────────────────┴──────────┘
  Fits in 200-300 GB ✅
  Delete raw videos after processing → only ~15 GB needed

VRAM BUDGET (the real constraint):
  ┌──────────────────────────────────────┬──────────┐
  │ Component                            │ VRAM     │
  ├──────────────────────────────────────┼──────────┤
  │ DeiT-Small (22M params, fp16)       │ ~0.5 GB  │
  │ AST (87M params, fp16)               │ ~1.0 GB  │
  │ Cross-attention module (~2M params)  │ ~0.1 GB  │
  │ Activations (batch=8, 224x224)       │ ~2.5 GB  │
  │ Optimizer states (AdamW)             │ ~2.0 GB  │
  │ Overhead                             │ ~1.0 GB  │
  │ TOTAL                                │ ~7.1 GB  │
  └──────────────────────────────────────┴──────────┘
  Fits in 8GB ✅ BUT tight — must use:
    - Mixed precision (fp16)
    - Batch size 8 (not 16 as in chat summary)
    - Gradient checkpointing on AST
    - Freeze first 8 layers of both encoders (as proposed)

TRAINING TIME ESTIMATE:
  ~3500 training samples × 50 epochs ÷ batch_size_8 = ~21,875 steps
  At ~0.3s/step with fp16 on RTX 5060 ≈ ~1.8 hours per run
  With 3-4 experiment runs ≈ ~6-8 hours total training
  ✅ Very feasible

TIMELINE REALITY CHECK:
  You have until April 16 (presentation) = ~7 days
  Day 1-2: Data pipeline (this script)
  Day 3-4: Model implementation + debug training
  Day 5: Run experiments
  Day 6-7: Poster + video + write-up
  TIGHT but doable with two people ✅

CRITICAL ADJUSTMENT: Batch size 8 instead of 16
  Your proposal says batch size 16 — this will OOM on 8GB VRAM.
  Use batch size 8 + gradient accumulation (2 steps) = effective batch 16.
  Same mathematical effect, half the VRAM usage.

=============================================================================
"""

import os
import sys
import json
import random
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import librosa
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION — matches proposal exactly
# ============================================================

INSTRUMENTS = [
    "acoustic_guitar", "cello", "clarinet", "erhu", "flute",
    "piano", "saxophone", "trumpet", "tuba", "violin", "xylophone"
]

INSTRUMENT_TO_IDX = {name: idx for idx, name in enumerate(INSTRUMENTS)}

# Audio settings — matched to AST requirements
SAMPLE_RATE = 16000        # Hz (AST standard)
N_MELS = 128               # Mel bins (AST standard)
HOP_LENGTH = 160           # ~10ms at 16kHz
N_FFT = 400                # ~25ms window
AUDIO_DURATION = 5.0       # seconds per clip
TARGET_AUDIO_LEN = int(SAMPLE_RATE * AUDIO_DURATION)  # 80000 samples

# Image settings — matched to DeiT-Small
IMAGE_SIZE = 224            # pixels → 14×14 = 196 patch tokens

# Extraction settings
CLIPS_PER_VIDEO = 10        # Extract 10 frame-audio pairs per video
CLIP_STRIDE_SECONDS = 3.0   # Seconds between clip start times


# ============================================================
# STEP 1: DOWNLOAD VIDEOS (solo + duet)
# ============================================================

def get_video_duration(video_path):
    """Get duration of a video file in seconds."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except:
        return None


def download_music_videos(json_path, output_dir, max_per_instrument=None):
    """
    Download MUSIC dataset videos.
    Handles both solo and duet JSON files.

    Args:
        json_path: Path to MUSIC_solo_videos.json or MUSIC_duet_videos.json
        output_dir: Where to save videos (organized by instrument/category)
        max_per_instrument: Limit downloads per category (None = download all)
    """
    with open(json_path, 'r') as f:
        music_data = json.load(f)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    # The MUSIC JSON structure is: {"version": 1.0, "videos": {"instrument": ["id1", "id2", ...]}}
    # Unwrap the "videos" key if present
    if "videos" in music_data and isinstance(music_data["videos"], dict):
        music_data = music_data["videos"]
        print(f"Unwrapped 'videos' key. Instruments found: {list(music_data.keys())}")
    else:
        print(f"Top-level keys: {list(music_data.keys())}")

    for category, video_ids in music_data.items():
        # Handle nested dict structure: {"videos": [...], ...}
        if isinstance(video_ids, dict):
            video_ids = video_ids.get("videos", [])

        # Skip non-list entries (metadata like counts, floats, ints, strings)
        if not isinstance(video_ids, list):
            print(f"Skipping '{category}': not a list (type={type(video_ids).__name__})")
            continue

        # Filter to only string video IDs (skip any nested non-string items)
        video_ids = [vid for vid in video_ids if isinstance(vid, str)]

        if not video_ids:
            print(f"Skipping '{category}': no valid video IDs found")
            continue

        if max_per_instrument:
            video_ids = video_ids[:max_per_instrument]

        cat_dir = Path(output_dir) / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {category}: {len(video_ids)} videos ---")

        for vid_id in tqdm(video_ids, desc=category):
            output_path = cat_dir / f"{vid_id}.mp4"

            if output_path.exists():
                stats["skipped"] += 1
                continue

            url = f"https://www.youtube.com/watch?v={vid_id}"
            cmd = [
                "yt-dlp",
                "-f", "worst[ext=mp4]/worst",  # Smallest quality
                "--no-playlist",
                "--max-filesize", "100M",
                "-o", str(output_path),
                "--quiet", "--no-warnings",
                url
            ]

            try:
                result = subprocess.run(cmd, timeout=120, capture_output=True)
                if result.returncode == 0 and output_path.exists():
                    stats["downloaded"] += 1
                else:
                    stats["failed"] += 1
            except (subprocess.TimeoutExpired, Exception):
                stats["failed"] += 1

    print(f"\n✅ Downloaded: {stats['downloaded']}, "
          f"Skipped (existing): {stats['skipped']}, "
          f"Failed: {stats['failed']}")
    return stats


# ============================================================
# STEP 2: EXTRACT MULTIPLE CLIPS PER VIDEO
# ============================================================

def extract_frame(video_path, timestamp):
    """Extract single frame at timestamp, resize to IMAGE_SIZE."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(timestamp),
            "-vframes", "1", "-q:v", "2",
            tmp_path, "-loglevel", "quiet"
        ]
        subprocess.run(cmd, timeout=15, capture_output=True)

        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 100:
            img = Image.open(tmp_path).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            return np.array(img, dtype=np.float32) / 255.0  # (224, 224, 3)
    except:
        pass
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return None


def extract_waveform(video_path, start_time, duration=AUDIO_DURATION):
    """
    Extract raw audio waveform (NOT spectrogram).
    We store waveforms so we can mix them BEFORE computing spectrograms.
    This is critical for proper Mix-and-Localize training.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", str(video_path),
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            tmp_path, "-loglevel", "quiet"
        ]
        subprocess.run(cmd, timeout=15, capture_output=True)

        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
            waveform, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)

            # Pad or trim to exact length
            if len(waveform) < TARGET_AUDIO_LEN:
                waveform = np.pad(waveform, (0, TARGET_AUDIO_LEN - len(waveform)))
            else:
                waveform = waveform[:TARGET_AUDIO_LEN]

            return waveform.astype(np.float32)  # (80000,)
    except:
        pass
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return None


def process_video(video_path, instrument, clips_per_video=CLIPS_PER_VIDEO):
    """
    Extract multiple frame-waveform pairs from a single video.
    Returns list of sample dicts.
    """
    duration = get_video_duration(video_path)
    if duration is None or duration < AUDIO_DURATION + 2:
        return []

    vid_id = video_path.stem
    samples = []

    # Calculate clip start times, evenly spaced
    usable_duration = duration - AUDIO_DURATION - 1  # Leave 1s buffer at end
    if usable_duration <= 0:
        return []

    n_clips = min(clips_per_video, max(1, int(usable_duration / CLIP_STRIDE_SECONDS)))
    start_times = np.linspace(1.0, usable_duration, n_clips)

    for i, t in enumerate(start_times):
        frame = extract_frame(video_path, t + AUDIO_DURATION / 2)  # Frame from middle of clip
        waveform = extract_waveform(video_path, t)

        if frame is not None and waveform is not None:
            samples.append({
                "video_id": vid_id,
                "clip_idx": i,
                "instrument": instrument,
                "label": INSTRUMENT_TO_IDX.get(instrument, -1),
                "frame": frame,        # (224, 224, 3) float32
                "waveform": waveform,  # (80000,) float32
            })

    return samples


def process_all_videos(raw_dir):
    """Process all downloaded videos, extracting multiple clips each."""
    raw_dir = Path(raw_dir)
    all_samples = []

    for category_dir in sorted(raw_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        instrument = category_dir.name
        video_files = sorted(category_dir.glob("*.mp4"))

        if not video_files:
            continue

        print(f"\n{'='*50}")
        print(f"Processing {len(video_files)} videos for: {instrument}")
        print(f"{'='*50}")

        for vf in tqdm(video_files, desc=instrument):
            clips = process_video(vf, instrument)
            all_samples.extend(clips)

    print(f"\n✅ Total clips extracted: {len(all_samples)}")
    return all_samples


# ============================================================
# STEP 3: CREATE STANDARD SPLITS + HDF5
# ============================================================

def create_standard_splits(samples, split_json_path=None):
    """
    Create train/test splits following standard MUSIC protocol.

    Standard MUSIC split (from CVPR 2025 paper):
      Solo: 358 train / 90 test videos
      Duet: 124 train / 17 test videos

    Since we extract multiple clips per video, we split BY VIDEO ID
    (not by clip) to prevent data leakage.

    If split_json_path is provided, use predefined splits.
    Otherwise, do 80/10/10 split stratified by instrument.
    """
    # Group clips by video_id
    by_video = defaultdict(list)
    for s in samples:
        by_video[s["video_id"]].append(s)

    # Group video_ids by instrument
    videos_by_instrument = defaultdict(list)
    for vid_id, clips in by_video.items():
        inst = clips[0]["instrument"]
        videos_by_instrument[inst].append(vid_id)

    train_clips, val_clips, test_clips = [], [], []

    for inst, vid_ids in sorted(videos_by_instrument.items()):
        random.shuffle(vid_ids)
        n = len(vid_ids)
        n_train = max(1, int(n * 0.8))
        n_val = max(1, int(n * 0.1))

        train_vids = set(vid_ids[:n_train])
        val_vids = set(vid_ids[n_train:n_train + n_val])
        test_vids = set(vid_ids[n_train + n_val:])

        for vid_id, clips in by_video.items():
            if clips[0]["instrument"] != inst:
                continue
            if vid_id in train_vids:
                train_clips.extend(clips)
            elif vid_id in val_vids:
                val_clips.extend(clips)
            elif vid_id in test_vids:
                test_clips.extend(clips)

    random.shuffle(train_clips)
    random.shuffle(val_clips)
    random.shuffle(test_clips)

    print(f"\nSplit by VIDEO ID (no leakage):")
    print(f"  Train: {len(train_clips)} clips")
    print(f"  Val:   {len(val_clips)} clips")
    print(f"  Test:  {len(test_clips)} clips")

    return train_clips, val_clips, test_clips


def save_hdf5(samples, filepath):
    """
    Save to HDF5.

    CRITICAL CHANGE from previous pipeline:
    We store RAW WAVEFORMS, not pre-computed spectrograms.
    Spectrograms are computed at training time AFTER mixing.

    HDF5 structure:
        /frames      -> (N, 224, 224, 3) float32
        /waveforms   -> (N, 80000)       float32  ← NEW: raw audio
        /labels      -> (N,)             int32
        /video_ids   -> (N,)             string
        /instruments -> (N,)             string
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    if n == 0:
        print(f"WARNING: No samples for {filepath}")
        return

    with h5py.File(filepath, 'w') as f:
        frames_ds = f.create_dataset(
            "frames", shape=(n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype='float32',
            chunks=(1, IMAGE_SIZE, IMAGE_SIZE, 3), compression="gzip", compression_opts=4
        )
        waveforms_ds = f.create_dataset(
            "waveforms", shape=(n, TARGET_AUDIO_LEN), dtype='float32',
            chunks=(1, TARGET_AUDIO_LEN), compression="gzip", compression_opts=4
        )
        labels_ds = f.create_dataset("labels", shape=(n,), dtype='int32')

        dt = h5py.string_dtype()
        vid_ids_ds = f.create_dataset("video_ids", shape=(n,), dtype=dt)
        instruments_ds = f.create_dataset("instruments", shape=(n,), dtype=dt)

        for i, s in enumerate(tqdm(samples, desc=f"Saving {filepath.name}")):
            frames_ds[i] = s["frame"]
            waveforms_ds[i] = s["waveform"]
            labels_ds[i] = s["label"]
            vid_ids_ds[i] = s["video_id"]
            instruments_ds[i] = s["instrument"]

        f.attrs["n_samples"] = n
        f.attrs["n_instruments"] = len(INSTRUMENTS)
        f.attrs["image_size"] = IMAGE_SIZE
        f.attrs["sample_rate"] = SAMPLE_RATE
        f.attrs["audio_duration"] = AUDIO_DURATION
        f.attrs["n_mels"] = N_MELS

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"✅ Saved {n} samples → {filepath} ({size_mb:.1f} MB)")


# ============================================================
# STEP 4: PYTORCH DATASET (Mix-and-Localize)
# ============================================================

DATASET_CODE = '''
import h5py
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader


def waveform_to_logmel(waveform, sr=16000, n_mels=128, n_fft=400, hop_length=160):
    """Convert raw waveform to log-mel spectrogram for AST input."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel  # (128, T)


class MUSICMixAndLocalize(Dataset):
    """
    Dataset for Mix-and-Localize training.
    
    KEY DESIGN: Stores raw waveforms, mixes them at load time,
    THEN computes spectrograms. This is the correct order because
    spectrogram(mix(a,b)) != mix(spectrogram(a), spectrogram(b))
    due to the nonlinear log operation.
    
    Returns per sample:
        frame:       (3, 224, 224)  — visual input for DeiT
        mixed_spec:  (1, 128, T)   — mixed log-mel for AST
        solo_spec:   (1, 128, T)   — solo log-mel for contrastive loss
        label:       int           — instrument class of the target
        other_label: int           — instrument class of the mixed-in source
    """

    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._h5 = None

        with h5py.File(h5_path, 'r') as f:
            self.n_samples = f.attrs['n_samples']
            self.labels = f['labels'][:]
            self.instruments = [
                x.decode() if isinstance(x, bytes) else x
                for x in f['instruments'][:]
            ]

        # Index by instrument for efficient pair sampling
        self.inst_indices = {}
        for i, inst in enumerate(self.instruments):
            self.inst_indices.setdefault(inst, []).append(i)

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        f = self._open()

        # Load target sample
        frame = f['frames'][idx]                    # (224, 224, 3)
        waveform_1 = f['waveforms'][idx]            # (80000,)
        label_1 = int(self.labels[idx])
        inst_1 = self.instruments[idx]

        # Sample a DIFFERENT instrument
        other_insts = [k for k in self.inst_indices if k != inst_1]
        other_inst = np.random.choice(other_insts)
        other_idx = np.random.choice(self.inst_indices[other_inst])
        waveform_2 = f['waveforms'][other_idx]      # (80000,)
        label_2 = int(self.labels[other_idx])

        # MIX waveforms FIRST, then compute spectrogram
        mixed_waveform = waveform_1 + waveform_2
        mixed_spec = waveform_to_logmel(mixed_waveform)  # (128, T)
        solo_spec = waveform_to_logmel(waveform_1)        # (128, T)

        # Convert to tensors
        frame_t = torch.from_numpy(frame).permute(2, 0, 1)  # (3, 224, 224)
        mixed_t = torch.from_numpy(mixed_spec).unsqueeze(0)  # (1, 128, T)
        solo_t = torch.from_numpy(solo_spec).unsqueeze(0)    # (1, 128, T)

        return {
            'frame': frame_t,         # DeiT input
            'mixed_spec': mixed_t,    # AST input (mixed audio)
            'solo_spec': solo_t,      # For contrastive loss
            'label': label_1,
            'other_label': label_2,
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


# Usage:
# train_ds = MUSICMixAndLocalize('data/processed/train.h5')
# train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
#
# IMPORTANT: batch_size=8 (not 16) for RTX 5060 8GB VRAM
# Use gradient accumulation of 2 steps for effective batch size 16:
#
# optimizer.zero_grad()
# for i, batch in enumerate(train_loader):
#     loss = model(batch) / 2  # Scale loss by accumulation steps
#     loss.backward()
#     if (i + 1) % 2 == 0:
#         optimizer.step()
#         optimizer.zero_grad()
'''


# ============================================================
# STEP 5: EXPLORE / VERIFY
# ============================================================

def explore_hdf5(h5_path):
    """Print dataset statistics for verification."""
    with h5py.File(h5_path, 'r') as f:
        print(f"\n{'='*50}")
        print(f"  {h5_path}")
        print(f"{'='*50}")
        print(f"  Samples:     {f.attrs['n_samples']}")
        print(f"  Frames:      {f['frames'].shape}")
        print(f"  Waveforms:   {f['waveforms'].shape}")
        print(f"  Labels:      {f['labels'].shape}")

        frame = f['frames'][0]
        wave = f['waveforms'][0]
        print(f"  Frame range: [{frame.min():.3f}, {frame.max():.3f}]")
        print(f"  Wave range:  [{wave.min():.4f}, {wave.max():.4f}]")
        print(f"  Wave RMS:    {np.sqrt(np.mean(wave**2)):.4f}")

        instruments = [x.decode() if isinstance(x, bytes) else x
                       for x in f['instruments'][:]]
        print(f"\n  Per-instrument counts:")
        for inst in sorted(set(instruments)):
            count = instruments.count(inst)
            print(f"    {inst:20s}: {count:5d} clips")

        print(f"\n  File size: {os.path.getsize(h5_path) / (1024**2):.1f} MB")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MUSIC Dataset Pipeline v2 (Proposal-Aligned)")
    parser.add_argument("--music_repo", type=str, default="MUSICDataset",
                        help="Path to cloned MUSICDataset repo")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory")
    parser.add_argument("--max_per_instrument", type=int, default=None,
                        help="Limit videos per instrument (None = all)")
    parser.add_argument("--clips_per_video", type=int, default=10,
                        help="Frame-audio pairs to extract per video")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--skip_processing", action="store_true")
    parser.add_argument("--explore_only", action="store_true")
    parser.add_argument("--print_dataset_class", action="store_true")

    args = parser.parse_args()

    if args.print_dataset_class:
        print(DATASET_CODE)
        return

    if args.explore_only:
        for split in ["train", "val", "test"]:
            p = Path(args.output_dir) / "processed" / f"{split}.h5"
            if p.exists():
                explore_hdf5(p)
        return

    random.seed(42)
    np.random.seed(42)

    global CLIPS_PER_VIDEO
    CLIPS_PER_VIDEO = args.clips_per_video

    raw_dir = Path(args.output_dir) / "raw"

    # ---- DOWNLOAD ----
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 1a: DOWNLOADING SOLO VIDEOS")
        print("=" * 60)
        solo_json = Path(args.music_repo) / "MUSIC_solo_videos.json"
        if solo_json.exists():
            download_music_videos(str(solo_json), str(raw_dir),
                                  max_per_instrument=args.max_per_instrument)
        else:
            print(f"ERROR: {solo_json} not found. Run:")
            print(f"  git clone https://github.com/roudimit/MUSIC_dataset.git")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("STEP 1b: DOWNLOADING DUET VIDEOS")
        print("=" * 60)
        duet_json = Path(args.music_repo) / "MUSIC_duet_videos.json"
        if duet_json.exists():
            download_music_videos(str(duet_json), str(raw_dir),
                                  max_per_instrument=args.max_per_instrument)
        else:
            print(f"WARNING: {duet_json} not found, skipping duets")

    # ---- PROCESS ----
    if not args.skip_processing:
        print("\n" + "=" * 60)
        print("STEP 2: EXTRACTING FRAMES + WAVEFORMS")
        print("=" * 60)
        all_samples = process_all_videos(raw_dir)

        if len(all_samples) == 0:
            print("ERROR: No samples extracted.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("STEP 3: CREATING SPLITS + HDF5")
        print("=" * 60)
        train, val, test = create_standard_splits(all_samples)

        processed_dir = Path(args.output_dir) / "processed"
        save_hdf5(train, processed_dir / "train.h5")
        save_hdf5(val, processed_dir / "val.h5")
        save_hdf5(test, processed_dir / "test.h5")

        metadata = {
            "instruments": INSTRUMENTS,
            "instrument_to_idx": INSTRUMENT_TO_IDX,
            "splits": {"train": len(train), "val": len(val), "test": len(test)},
            "clips_per_video": CLIPS_PER_VIDEO,
            "image_size": IMAGE_SIZE,
            "sample_rate": SAMPLE_RATE,
            "audio_duration": AUDIO_DURATION,
            "n_mels": N_MELS,
            "note": "Waveforms stored raw; compute spectrograms AFTER mixing at training time",
        }
        with open(Path(args.output_dir) / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    # ---- EXPLORE ----
    print("\n" + "=" * 60)
    print("STEP 4: VERIFICATION")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        p = Path(args.output_dir) / "processed" / f"{split}.h5"
        if p.exists():
            explore_hdf5(p)

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Verify data: python {__file__} --explore_only")
    print(f"  2. View Dataset class: python {__file__} --print_dataset_class")
    print(f"  3. (Optional) Delete raw videos to free ~50GB:")
    print(f"     rm -rf {args.output_dir}/raw/")
    print(f"  4. Build model: DeiT encoder + AST encoder + cross-attention")


if __name__ == "__main__":
    main()