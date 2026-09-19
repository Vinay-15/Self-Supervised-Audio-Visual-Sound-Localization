import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
evaluate.py — Evaluation + heatmap visualization.

Metrics (from proposal Section 4):
    - cIoU: Threshold heatmap at 0.5, compute IoU with ground truth bbox
    - AUC: Pixel-level ROC across all thresholds

Usage:
    python evaluate.py --data_dir ./data --checkpoint ./checkpoints/best_model.pt
    python evaluate.py --data_dir ./data --checkpoint ./checkpoints/best_model.pt --visualize
"""

import json
import argparse
from pathlib import Path

import numpy as np
import h5py
import librosa
import torch
import torch.nn.functional as F
# ── FIX: use the non-deprecated autocast import
from torch.amp import autocast
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── FIX: correct class name (was SoundLocalizationModel)
from model import SoundSourceLocalizer


def waveform_to_logmel(waveform, sr=16000, n_mels=128, n_fft=400, hop_length=160):
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    return librosa.power_to_db(mel, ref=np.max)


def generate_heatmap(model, frame, spectrogram, device):
    model.eval()
    with torch.no_grad():
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
        spec_t = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(device)
        # ── FIX: updated autocast call + unpack only 2 values we need
        with autocast('cuda', dtype=torch.float16):
            heatmap, _, _, _ = model(frame_t, spec_t)
        heatmap = heatmap.squeeze().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def compute_ciou(heatmap, gt_bbox, threshold=0.5):
    pred_mask = (heatmap >= threshold).astype(float)
    gt_mask = np.zeros_like(heatmap)
    x1, y1, x2, y2 = gt_bbox
    gt_mask[y1:y2, x1:x2] = 1.0
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    return float(intersection / union) if union > 1e-6 else 0.0


def compute_auc(heatmap, gt_bbox):
    gt_mask = np.zeros_like(heatmap)
    x1, y1, x2, y2 = gt_bbox
    gt_mask[y1:y2, x1:x2] = 1.0
    gt_flat = gt_mask.flatten()
    pred_flat = heatmap.flatten()
    if gt_flat.sum() == 0 or gt_flat.sum() == len(gt_flat):
        return 0.5
    try:
        return roc_auc_score(gt_flat, pred_flat)
    except:
        return 0.5


def evaluate_synthetic(model, h5_path, device, max_samples=100):
    print(f"\n  Evaluating on {Path(h5_path).name}...")
    with h5py.File(h5_path, 'r') as f:
        n = min(f.attrs['n_samples'], max_samples)
        instruments = [x.decode() if isinstance(x, bytes) else x for x in f['instruments'][:n]]
        ciou_all, auc_all = [], []
        per_inst = {}

        for i in range(n):
            frame = f['frames'][i]
            spec = waveform_to_logmel(f['waveforms'][i])
            heatmap = generate_heatmap(model, frame, spec, device)
            gt_bbox = (45, 45, 179, 179)
            ciou = compute_ciou(heatmap, gt_bbox)
            auc = compute_auc(heatmap, gt_bbox)
            ciou_all.append(ciou)
            auc_all.append(auc)
            inst = instruments[i]
            per_inst.setdefault(inst, {'ciou': [], 'auc': []})
            per_inst[inst]['ciou'].append(ciou)
            per_inst[inst]['auc'].append(auc)

    print(f"    Overall (n={n}): cIoU={np.mean(ciou_all):.4f}  AUC={np.mean(auc_all):.4f}")
    print(f"    Per-instrument:")
    for inst in sorted(per_inst):
        c = np.mean(per_inst[inst]['ciou'])
        a = np.mean(per_inst[inst]['auc'])
        print(f"      {inst:20s}: cIoU={c:.4f}  AUC={a:.4f}  (n={len(per_inst[inst]['ciou'])})")

    return {'ciou': float(np.mean(ciou_all)), 'auc': float(np.mean(auc_all)),
            'per_instrument': {k: {'ciou': float(np.mean(v['ciou'])), 'auc': float(np.mean(v['auc']))}
                               for k, v in per_inst.items()}}


def visualize_heatmaps(model, h5_path, device, output_dir, n_samples=12):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        n = min(f.attrs['n_samples'], n_samples)
        instruments = [x.decode() if isinstance(x, bytes) else x for x in f['instruments'][:n]]

        # Grid view
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        # ── FIX: flatten axes array safely regardless of rows/cols shape
        axes_flat = np.array(axes).reshape(-1).tolist()

        for i in range(n):
            frame = f['frames'][i]
            spec = waveform_to_logmel(f['waveforms'][i])
            heatmap = generate_heatmap(model, frame, spec, device)
            ax = axes_flat[i]
            ax.imshow(frame)
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            ax.set_title(instruments[i], fontsize=10)
            ax.axis('off')
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.suptitle("Sound source localization heatmaps", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "heatmaps_grid.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Individual detailed views
        for i in range(min(n, 6)):
            frame = f['frames'][i]
            spec = waveform_to_logmel(f['waveforms'][i])
            heatmap = generate_heatmap(model, frame, spec, device)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            ax1.imshow(frame); ax1.set_title(f"Frame ({instruments[i]})"); ax1.axis('off')
            ax2.imshow(heatmap, cmap='jet'); ax2.set_title("Heatmap"); ax2.axis('off')
            ax3.imshow(frame); ax3.imshow(heatmap, cmap='jet', alpha=0.5); ax3.set_title("Overlay"); ax3.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / f"heatmap_{i}_{instruments[i]}.png", dpi=150, bbox_inches='tight')
            plt.close()

    print(f"  Saved visualizations to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/best_model.pt")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── FIX: correct class name (was SoundLocalizationModel)
    model = SoundSourceLocalizer().to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"WARNING: No checkpoint found, using random weights")

    test_h5 = str(Path(args.data_dir) / "processed" / "test.h5")
    if os.path.exists(test_h5):
        results = evaluate_synthetic(model, test_h5, device, args.max_samples)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "eval_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        if args.visualize:
            visualize_heatmaps(model, test_h5, device, args.output_dir)
    else:
        print(f"  {test_h5} not found!")


if __name__ == "__main__":
    main()
