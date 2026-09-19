"""
model.py — Fully Transformer-based Sound Source Localization
Architecture: DeiT-Small + ViT-Small(AST) + Cross-Modal Attention
NO ResNet. NO VGGish.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DeiTVisualEncoder(nn.Module):
    """
    DeiT-Small pretrained on ImageNet.
    Input:  (B, 3, 224, 224)
    Output: (B, 196, 384) — 14x14 spatial patch tokens
    Freeze first 8 of 12 layers.
    """

    def __init__(self, freeze_layers=8):
        super().__init__()
        self.deit = timm.create_model('deit_small_patch16_224', pretrained=True)
        self.deit.head = nn.Identity()

        for param in self.deit.patch_embed.parameters():
            param.requires_grad = False
        for i, block in enumerate(self.deit.blocks):
            if i < freeze_layers:
                for param in block.parameters():
                    param.requires_grad = False

        self.embed_dim = 384

    def forward(self, x):
        B = x.shape[0]
        x = self.deit.patch_embed(x)
        cls_token = self.deit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.deit.pos_embed
        x = self.deit.pos_drop(x)
        for block in self.deit.blocks:
            x = block(x)
        x = self.deit.norm(x)
        patch_tokens = x[:, 1:, :]
        return patch_tokens


class ASTAudioEncoder(nn.Module):
    """
    ViT-Small treating spectrogram as image (AST approach).
    Input:  (B, 1, 128, T) — log-mel spectrogram
    Output: (B, 768) — global audio embedding
    """

    def __init__(self, embed_dim=768, freeze_layers=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.vit = timm.create_model(
            'vit_small_patch16_224', pretrained=True,
            in_chans=1, num_classes=0, global_pool='token'
        )
        self.proj = nn.Linear(384, embed_dim)

        for param in self.vit.patch_embed.parameters():
            param.requires_grad = False
        for i, block in enumerate(self.vit.blocks):
            if i < freeze_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, spec):
        spec = F.interpolate(spec, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.vit(spec)
        audio_embed = self.proj(features)
        return audio_embed


class CrossModalAttention(nn.Module):
    """
    Audio queries attend over visual spatial tokens.
    Input:  visual_tokens (B, 196, 384), audio_embed (B, 768)
    Output: heatmap (B, 1, 224, 224), attn_weights (B, 196)
    """

    def __init__(self, visual_dim=384, audio_dim=768, proj_dim=256):
        super().__init__()
        self.proj_dim = proj_dim
        self.W_Q = nn.Linear(audio_dim, proj_dim)
        self.W_K = nn.Linear(visual_dim, proj_dim)
        self.W_V = nn.Linear(visual_dim, proj_dim)
        self.scale = proj_dim ** 0.5

    def forward(self, visual_tokens, audio_embed):
        B = visual_tokens.shape[0]
        Q = self.W_Q(audio_embed).unsqueeze(1)
        K = self.W_K(visual_tokens)
        V = self.W_V(visual_tokens)

        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)

        attn_map = attn_weights.squeeze(1).view(B, 1, 14, 14)
        heatmap = F.interpolate(attn_map, size=(224, 224),
                                mode='bilinear', align_corners=False)
        return heatmap, attn_weights.squeeze(1)


class SoundSourceLocalizer(nn.Module):
    """Full model: DeiT + AST + CrossAttention → heatmap."""

    def __init__(self, freeze_layers=8, proj_dim=256):
        super().__init__()
        self.visual_encoder = DeiTVisualEncoder(freeze_layers=freeze_layers)
        self.audio_encoder = ASTAudioEncoder(freeze_layers=freeze_layers)
        self.cross_attention = CrossModalAttention(
            visual_dim=self.visual_encoder.embed_dim,
            audio_dim=self.audio_encoder.embed_dim,
            proj_dim=proj_dim,
        )

    # ── FIX: return all 4 values so train.py can reuse them ──────────────────
    def forward(self, frame, spectrogram):
        visual_tokens = self.visual_encoder(frame)
        audio_embed = self.audio_encoder(spectrogram)
        heatmap, attn_weights = self.cross_attention(visual_tokens, audio_embed)
        return heatmap, attn_weights, visual_tokens, audio_embed
    # ─────────────────────────────────────────────────────────────────────────

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params:    {total - trainable:,}")
        return total, trainable


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SoundSourceLocalizer(freeze_layers=8).to(device)
    model.count_params()

    frame = torch.randn(2, 3, 224, 224).to(device)
    spec = torch.randn(2, 1, 128, 501).to(device)

    with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        heatmap, attn, vis_tok, aud_emb = model(frame, spec)

    print(f"\nInput frame:    {frame.shape}")
    print(f"Input spec:     {spec.shape}")
    print(f"Output heatmap: {heatmap.shape}")
    print(f"Output attn:    {attn.shape}")
    print(f"Output vis_tok: {vis_tok.shape}")
    print(f"Output aud_emb: {aud_emb.shape}")

    if torch.cuda.is_available():
        print(f"\nGPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
