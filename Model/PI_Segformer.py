# ============================================
# FINAL VERSION: Loss + Train/Val mIoU + Dice
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import SegformerConfig, SegformerModel
# =====================================================
# Fourier Encoding
# =====================================================
def fourier_encode(x, num_freqs=4):
    out = [x]
    for i in range(num_freqs):
        out.append(torch.sin(2**i * x))
        out.append(torch.cos(2**i * x))
    return torch.cat(out, dim=-1)


# =====================================================
# Attention & FiLM
# =====================================================
class ChannelAttention(nn.Module):
    def __init__(self, C, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(C, C // r),
            nn.ReLU(),
            nn.Linear(C // r, C)
        )

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3))
        att = torch.sigmoid(self.mlp(avg))
        return x * att.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class FiLM(nn.Module):
    def forward(self, feat, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, feat_flat, cond_feat, H, W):
        cond_token = cond_feat.unsqueeze(1)

        attn_out, attn_weights = self.attn(
            cond_token, feat_flat, feat_flat
        )

        attn_map = attn_weights.view(feat_flat.shape[0], 1, H, W)
        return attn_map
# =====================================================
# Condition Encoder
# =====================================================
class CondsEncoder(nn.Module):
    def __init__(self, out_dim, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, conds):
        return self.mlp(conds)

# =====================================================
# Backbone
# =====================================================

class Backbone(nn.Module):
    def __init__(self, in_ch=1, out_ch=128):
        super().__init__()

        # SegFormer encoder
        config = SegformerConfig(
            hidden_sizes=[64, 128, 320, 512],
            depths=[3, 4, 6, 3],
            num_attention_heads=[1, 2, 5, 8]
        )

        self.model = SegformerModel(config)

        # 改输入通道（1 → 3）
        if in_ch == 1:
            self.input_proj = nn.Conv2d(1, 3, kernel_size=1)
        else:
            self.input_proj = nn.Identity()

        # SegFormer输出通道
        self.proj = nn.ModuleList([
            nn.Conv2d(c, out_ch, 1) for c in [64, 128, 320, 512]
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        H, W = x.shape[-2:]

        x = self.input_proj(x)

        outputs = self.model(x, output_hidden_states=True)

        # 取4个stage特征
        feats = outputs.hidden_states[-4:]

        up_feats = []
        for i, f in enumerate(feats):
            f = self.proj[i](f)
            f = F.interpolate(f, size=(H, W), mode='bilinear', align_corners=False)
            up_feats.append(f)

        x = torch.cat(up_feats, dim=1)
        x = self.fuse(x)

        return x, feats[-1]   # 返回最后一层


# =====================================================
# Model
# =====================================================
class IRTDefectDetector(nn.Module):
    def __init__(self, in_ch=1, out_ch=128, num_freqs=4):
        super().__init__()

        self.num_freqs = num_freqs
        cond_dim = 3 * (1 + 2 * num_freqs)

        self.backbone = Backbone(in_ch, out_ch)

        self.conds_enc = CondsEncoder(out_ch, cond_dim)
        self.cross_attn = CrossAttentionFusion(out_ch)

        self.head = nn.Conv2d(out_ch, 1, 1)
        nn.init.constant_(self.head.bias, -3.0)
        self.low_proj = nn.Conv2d(512, 128, 1)

    def forward(self, x_img, conds):
        conds = fourier_encode(conds, self.num_freqs)

        # ========= backbone =========
        feat, feat_low = self.backbone(x_img)

        # ========= cond =========
        cond_feat = self.conds_enc(conds)  # [B,128]

        # ========= low-res cross attention =========

        # 通道对齐（512→128）
        feat_low = self.low_proj(feat_low)

        B, C, H, W = feat_low.shape  # H=20, W=15

        # flatten
        feat_flat = feat_low.flatten(2).permute(0, 2, 1)  # [B,300,128]

        # reshape → 空间图
        attn_map = self.cross_attn(feat_flat, cond_feat, H, W)
        # 上采样到原图
        attn_map = F.interpolate(attn_map, size=(480, 640), mode='bilinear', align_corners=False)
        # 融合
        attn_map = torch.sigmoid(attn_map)
        feat = feat * (1 + attn_map)

        # ========= head =========
        return torch.sigmoid(self.head(feat))


# =====================================================
# Dataset
# =====================================================
class IRTPDataset(Dataset):
    def __init__(self, npz_folder):

        self.temps, self.labels, self.conds = [], [], []

        files = sorted([
            os.path.join(npz_folder, f)
            for f in os.listdir(npz_folder)
            if f.endswith(".npz")
        ])

        for f in files:
            data = np.load(f)

            temp = data["temps"].astype(np.float32)
            if temp.ndim == 3:
                temp = temp[:, None, :, :]
            self.temps.append(temp)

            label = data["labels"].astype(np.float32)
            if label.ndim == 3:
                label = label[:, None, :, :]
            self.labels.append(label)

            cond = data["conds"].astype(np.float32)

            cond[:, 0] /= 300.0
            cond[:, 1] /= 200.0

            self.conds.append(cond)

        self.temps = np.concatenate(self.temps)
        self.labels = np.concatenate(self.labels)
        self.conds = np.concatenate(self.conds)

        print(f"Loaded samples: {len(self.temps)}")

    def __len__(self):
        return len(self.temps)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.temps[idx]),
            torch.tensor(self.conds[idx]),
            torch.tensor(self.labels[idx]),
        )


# =====================================================
# Loss
# =====================================================
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def total_loss(pred, target):
    return F.binary_cross_entropy(pred, target) + dice_loss(pred, target)


# =====================================================
# Metrics
# =====================================================
def compute_miou(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = ((pred + target) > 0).float().sum(dim=(1,2,3))

    return (intersection / (union + 1e-8)).mean().item()


def compute_dice(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    return (2 * intersection / (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-8)).mean().item()


# =====================================================
# Training
# =====================================================
if __name__ == "__main__":

    dataset = IRTPDataset(r"..\Dataset")

    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    train_loader = DataLoader(dataset, batch_size=2,
                              sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=2,
                            sampler=torch.utils.data.SubsetRandomSampler(val_idx))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IRTDefectDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 200

    train_losses, val_losses = [], []
    train_mious, val_mious = [], []
    train_dices, val_dices = [], []

    for epoch in range(num_epochs):

        # ================= TRAIN =================
        model.train()
        train_loss = 0
        train_miou_epoch = 0
        train_dice_epoch = 0
        train_count = 0

        for temp, cond, label in train_loader:
            temp, cond, label = temp.to(device), cond.to(device), label.to(device)

            pred = model(temp, cond)
            loss = total_loss(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_miou_epoch += compute_miou(pred, label)
            train_dice_epoch += compute_dice(pred, label)
            train_count += 1

        train_loss /= len(train_loader)
        train_miou_epoch /= train_count
        train_dice_epoch /= train_count

        train_losses.append(train_loss)
        train_mious.append(train_miou_epoch)
        train_dices.append(train_dice_epoch)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0
        val_miou_epoch = 0
        val_dice_epoch = 0
        val_count = 0

        with torch.no_grad():
            for temp, cond, label in val_loader:
                temp, cond, label = temp.to(device), cond.to(device), label.to(device)

                pred = model(temp, cond)
                loss = total_loss(pred, label)

                val_loss += loss.item()
                val_miou_epoch += compute_miou(pred, label)
                val_dice_epoch += compute_dice(pred, label)
                val_count += 1

        val_loss /= len(val_loader)
        val_miou_epoch /= val_count
        val_dice_epoch /= val_count

        val_losses.append(val_loss)
        val_mious.append(val_miou_epoch)
        val_dices.append(val_dice_epoch)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Train mIoU {train_miou_epoch:.4f} | Val mIoU {val_miou_epoch:.4f} | "
            f"Train Dice {train_dice_epoch:.4f} | Val Dice {val_dice_epoch:.4f}"
        )
        # ================= SAVE MODEL =================
        # best model
        if val_miou_epoch > best_miou:
            best_miou = val_miou_epoch

            torch.save(model.state_dict(), r"..\Weights\best_model.pth")
            print(f"Saved BEST model at epoch {epoch + 1}")

        # 每20个epoch保存
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), r"..\Weights\epoch_{epoch + 1}.pth")
            print(f"Saved checkpoint at epoch {epoch + 1}")
    # ================= SAVE =================
    df = pd.DataFrame({
        "Epoch": np.arange(1, num_epochs+1),
        "Train Loss": train_losses,
        "Val Loss": val_losses,
        "Train mIoU": train_mious,
        "Val mIoU": val_mious,
        "Train Dice": train_dices,
        "Val Dice": val_dices
    })

    save_path = r"..\Weights\metrics_full_delamination.xlsx"
    df.to_excel(save_path, index=False)

    print("Saved metrics to:", save_path)
    torch.save(model, r"..\Weights\model_delamination.pt")