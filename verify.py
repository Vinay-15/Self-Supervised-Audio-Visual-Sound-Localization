"""
verify_data.py — Run this to check your dataset is ready for training.

Usage:
    python verify_data.py --data_dir ./data
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import h5py

import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def check_raw_videos(data_dir):
    raw_dir = Path(data_dir) / "raw"
    if not raw_dir.exists():
        print("  No raw/ directory found (already deleted or not yet downloaded)")
        return

    print("\n  RAW VIDEOS:")
    total = 0
    for cat_dir in sorted(raw_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        videos = list(cat_dir.glob("*.mp4"))
        total += len(videos)
        sizes = [v.stat().st_size / (1024 * 1024) for v in videos]
        avg = sum(sizes) / len(sizes) if sizes else 0
        print(f"    {cat_dir.name:20s}: {len(videos):3d} videos, avg {avg:.1f} MB")
    print(f"    TOTAL: {total} videos")


def check_hdf5(h5_path):
    if not os.path.exists(h5_path):
        print(f"    MISSING: {h5_path}")
        return None

    with h5py.File(h5_path, 'r') as f:
        n = f.attrs.get('n_samples', 0)
        name = Path(h5_path).name
        print(f"\n  {name}: {n} samples")

        for key in ['frames', 'waveforms', 'labels', 'video_ids', 'instruments']:
            if key in f:
                print(f"    {key:15s}: {str(f[key].shape):20s} {f[key].dtype}")
            else:
                print(f"    {key:15s}: MISSING!")

        if n == 0:
            return None

        # Value ranges
        fr = f['frames'][0]
        wv = f['waveforms'][0]
        print(f"    Frame range:  [{fr.min():.3f}, {fr.max():.3f}]")
        print(f"    Wave RMS:     {np.sqrt(np.mean(wv ** 2)):.6f}")

        # Silent / blank check
        check_n = min(n, 20)
        silent = sum(1 for i in range(check_n) if np.sqrt(np.mean(f['waveforms'][i] ** 2)) < 1e-6)
        blank = sum(1 for i in range(check_n) if f['frames'][i].std() < 0.01)
        print(f"    Silent audio:  {silent}/{check_n}  {'BAD' if silent > check_n // 2 else 'OK'}")
        print(f"    Blank frames:  {blank}/{check_n}  {'BAD' if blank > check_n // 2 else 'OK'}")

        # Distribution
        instruments = [x.decode() if isinstance(x, bytes) else x for x in f['instruments'][:]]
        counts = Counter(instruments)
        print(f"    Instruments: {dict(sorted(counts.items()))}")
        return {'n_samples': n, 'instruments': counts}


def check_dataloader(h5_path):
    print(f"\n  DATALOADER TEST:")
    try:
        import torch
        import librosa
        from torch.utils.data import DataLoader

        class TestDS(torch.utils.data.Dataset):
            def __init__(self, path):
                self.path = path
                self._h5 = None
                with h5py.File(path, 'r') as f:
                    self.n = f.attrs['n_samples']
                    self.labels = f['labels'][:]
                    self.insts = [x.decode() if isinstance(x, bytes) else x for x in f['instruments'][:]]
                self.idx_map = {}
                for i, inst in enumerate(self.insts):
                    self.idx_map.setdefault(inst, []).append(i)

            def __len__(self): return self.n

            def _f(self):
                if self._h5 is None: self._h5 = h5py.File(self.path, 'r')
                return self._h5

            def __getitem__(self, idx):
                f = self._f()
                frame = f['frames'][idx]
                w1 = f['waveforms'][idx]
                other = [k for k in self.idx_map if k != self.insts[idx]]
                oi = np.random.choice(self.idx_map[np.random.choice(other)])
                w2 = f['waveforms'][oi]
                mixed = w1 + w2
                ms = librosa.power_to_db(librosa.feature.melspectrogram(y=mixed, sr=16000, n_mels=128, n_fft=400, hop_length=160), ref=np.max)
                ss = librosa.power_to_db(librosa.feature.melspectrogram(y=w1, sr=16000, n_mels=128, n_fft=400, hop_length=160), ref=np.max)
                return {
                    'frame': torch.from_numpy(frame).permute(2, 0, 1).float(),
                    'mixed_spec': torch.from_numpy(ms).unsqueeze(0).float(),
                    'solo_spec': torch.from_numpy(ss).unsqueeze(0).float(),
                    'label': int(self.labels[idx]),
                }

        ds = TestDS(h5_path)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        print(f"    frame:      {batch['frame'].shape}")
        print(f"    mixed_spec: {batch['mixed_spec'].shape}")
        print(f"    solo_spec:  {batch['solo_spec'].shape}")
        print(f"    labels:     {batch['label'].tolist()}")
        print(f"    PASSED")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()

    print("=" * 50)
    print("  DATASET VERIFICATION")
    print("=" * 50)
    check_raw_videos(args.data_dir)

    print("\n  HDF5 FILES:")
    total = 0
    for split in ["train", "val", "test"]:
        r = check_hdf5(str(Path(args.data_dir) / "processed" / f"{split}.h5"))
        if r: total += r['n_samples']

    train_h5 = str(Path(args.data_dir) / "processed" / "train.h5")
    if os.path.exists(train_h5):
        check_dataloader(train_h5)

    print(f"\n  TOTAL SAMPLES: {total}")
    if total >= 50:
        print("  STATUS: Ready for debug training")
    else:
        print("  STATUS: Too few — run pipeline first or download more videos")

if __name__ == "__main__":
    main()