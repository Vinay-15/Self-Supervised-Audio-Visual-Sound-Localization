import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
train.py — Training loop for Sound Source Localization
Loss = L_loc (KL solo→mixed) + 0.5 * L_contrast (InfoNCE)
Batch 8 + gradient accum 2 = effective 16. Mixed precision for 8GB VRAM.

CHANGES FROM ORIGINAL (3 only):
  1. LocalizationLoss replaced with MixLocalizeLoss.
       Old: entropy(heatmap_mixed) — stuck at log(50176)=10.823 forever.
       New: KL(solo_attn || mixed_attn) — spatial signal via solo_spec.
  2. solo_spec loaded from batch and passed to loc_fn.
       Old: solo_spec was computed in dataset but never used in train loop.
  3. ctr_fn parameters added to optimizer.
       Old: audio_proj trained with zero gradient updates (outside model).
"""

import time
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset.music_dataset import MUSICDataset
from model import SoundSourceLocalizer


# ── CHANGE 1: replace LocalizationLoss with MixLocalizeLoss ──────────────────
#
# WHY: The old entropy loss -(p log p).sum() starts at log(224*224)=10.823
# for a uniform heatmap and has NO signal for WHERE to peak, only that it
# should peak somewhere. So loc stayed at 10.823 all 50 epochs.
#
# FIX: Run a second forward pass on solo_spec (no grad, cheap). The solo
# attention map is a pseudo-GT spatial distribution. KL(solo||mixed) tells
# the model: "your attention on mixed audio should look like your attention
# on solo audio, because the target instrument is still in frame."
# This gives a real spatial gradient on every step.
# ─────────────────────────────────────────────────────────────────────────────
class MixLocalizeLoss(nn.Module):
    def forward(self, model, frame, mixed_spec, solo_spec):
        # mixed_spec heatmap — the one we want to improve (gradients flow)
        heatmap_mixed, _, _, _ = model(frame, mixed_spec)

        # solo_spec heatmap — pseudo ground truth (no gradients needed)
        with torch.no_grad():
            heatmap_solo, _, _, _ = model(frame, solo_spec)

        B = heatmap_mixed.shape[0]
        mixed_prob = F.softmax(heatmap_mixed.view(B, -1), dim=-1)
        solo_prob  = F.softmax(heatmap_solo.view(B, -1),  dim=-1)

        # KL(solo || mixed): mixed is our prediction, solo is the target
        return F.kl_div(torch.log(mixed_prob + 1e-8), solo_prob.detach(),
                        reduction='batchmean')


class ContrastiveLoss(nn.Module):
    """InfoNCE: matched audio-visual pairs closer, mismatched apart."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.audio_proj = nn.Linear(768, 384)

    def forward(self, audio_embed, visual_tokens, attn_weights):
        weighted_visual = torch.bmm(
            attn_weights.unsqueeze(1), visual_tokens
        ).squeeze(1)

        audio_norm  = F.normalize(audio_embed,    dim=-1)
        visual_norm = F.normalize(weighted_visual, dim=-1)
        audio_proj  = F.normalize(self.audio_proj(audio_norm), dim=-1)

        sim    = torch.mm(audio_proj, visual_norm.t()) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2


def train_one_epoch(model, loader, optimizer, scaler, device, loc_fn, ctr_fn, lam=0.5, accum=2):
    model.train()
    total_loss = total_loc = total_ctr = 0
    n = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        frame      = batch['frame'].to(device)
        mixed_spec = batch['mixed_spec'].to(device)
        solo_spec  = batch['solo_spec'].to(device)   # CHANGE 2: was ignored before

        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            # loc_fn now handles both forward passes internally
            l_loc = loc_fn(model, frame, mixed_spec, solo_spec)

            # re-use the mixed forward pass outputs for contrastive loss
            heatmap, attn, vis_tok, aud_emb = model(frame, mixed_spec)
            l_ctr = ctr_fn(aud_emb, vis_tok, attn)

            loss = (l_loc + lam * l_ctr) / accum

        scaler.scale(loss).backward()
        if (i + 1) % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum
        total_loc  += l_loc.item()
        total_ctr  += l_ctr.item()
        n += 1

    return {'loss': total_loss/max(n,1), 'loc': total_loc/max(n,1), 'ctr': total_ctr/max(n,1)}


@torch.no_grad()
def validate(model, loader, device, loc_fn, ctr_fn, lam=0.5):
    model.eval()
    total = 0; n = 0
    for batch in loader:
        frame      = batch['frame'].to(device)
        mixed_spec = batch['mixed_spec'].to(device)
        solo_spec  = batch['solo_spec'].to(device)

        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            l_loc = loc_fn(model, frame, mixed_spec, solo_spec)
            heatmap, attn, vis_tok, aud_emb = model(frame, mixed_spec)
            loss = l_loc + lam * ctr_fn(aud_emb, vis_tok, attn)
        total += loss.item(); n += 1
    return {'val_loss': total/max(n,1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/processed")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_contrast", type=float, default=0.5)
    parser.add_argument("--freeze_layers", type=int, default=8)
    parser.add_argument("--save_dir", default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    train_ds = MUSICDataset(os.path.join(args.data_dir, "train.h5"))
    val_ds   = MUSICDataset(os.path.join(args.data_dir, "val.h5"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")

    model = SoundSourceLocalizer(freeze_layers=args.freeze_layers).to(device)
    model.count_params()

    loc_fn = MixLocalizeLoss()
    ctr_fn = ContrastiveLoss().to(device)

    # CHANGE 3: ctr_fn.parameters() added so audio_proj actually trains
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(ctr_fn.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float('inf')
    history  = []

    print(f"\nTraining {args.epochs} epochs, effective batch {args.batch_size * args.accum_steps}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device,
                             loc_fn, ctr_fn, args.lambda_contrast, args.accum_steps)
        vl = validate(model, val_loader, device, loc_fn, ctr_fn, args.lambda_contrast)
        scheduler.step()
        elapsed = time.time() - t0

        lr = scheduler.get_last_lr()[0]
        history.append({**tr, **vl, 'epoch': epoch, 'lr': lr})

        print(f"Ep {epoch:3d}/{args.epochs} | "
              f"loss={tr['loss']:.4f} (loc={tr['loc']:.3f} ctr={tr['ctr']:.3f}) | "
              f"val={vl['val_loss']:.4f} | lr={lr:.1e} | {elapsed:.1f}s")

        if vl['val_loss'] < best_val:
            best_val = vl['val_loss']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': best_val}, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  -> Saved best (val={best_val:.4f})")

    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val: {best_val:.4f}. Saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
