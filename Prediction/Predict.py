# ============================================================
# Prediction Script (Clean + mIoU + Dice)
# ============================================================

import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

from Model.PI_Segformer import IRTDefectDetector

# ============================================================
# 配置
# ============================================================
model_path = r"..\Weights\epoch_200.pth"
excel_path = r"..\Dataset_making\Exceldata\TE_00681.xlsx"
npz_path   = r"..\Dataset\TE_00681.npz"

save_dir = r"..\Prediction\predict_results"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 条件参数
q_val = 75.75
delta_T_val = 60
wet_val = 0

threshold = 0.5
min_area = 150

# ============================================================
# 1. 读取温度图
# ============================================================
def load_excel_temperature(excel_path, img_size=(640, 480)):
    df = pd.read_excel(excel_path, header=None)
    temp_map = df.values.astype(np.float32)

    temp_map_resized = cv2.resize(temp_map, img_size)

    temp_map_norm = (temp_map_resized - temp_map_resized.min()) / \
                    (temp_map_resized.max() - temp_map_resized.min() + 1e-8)

    temp_map_norm = np.expand_dims(temp_map_norm, axis=(0, 1))
    return torch.tensor(temp_map_norm, dtype=torch.float32), temp_map_resized


# ============================================================
# 2. mask → boxes
# ============================================================
def generate_boxes(mask, min_area=150):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))

    return boxes


# ============================================================
# 3. mIoU / Dice
# ============================================================
def compute_miou(pred_mask, gt_mask):

    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return intersection / (union + 1e-8)


def compute_dice(pred_mask, gt_mask):

    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    intersection = (pred * gt).sum()

    return (2. * intersection) / (pred.sum() + gt.sum() + 1e-8)


# ============================================================
# 4. 加载模型
# ============================================================
model = IRTDefectDetector(in_ch=1, out_ch=128, num_freqs=4)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# ============================================================
# 5. 数据准备
# ============================================================
temp_tensor, temp_resized = load_excel_temperature(excel_path)
temp_tensor = temp_tensor.to(device)

# cond归一化
cond = torch.tensor([[q_val, delta_T_val, wet_val]], dtype=torch.float32, device=device)
cond[:, 0] /= 300.0
cond[:, 1] /= 200.0

# ============================================================
# 6. 预测
# ============================================================
with torch.no_grad():
    pred = model(temp_tensor, cond)
    pred_np = pred.cpu().numpy()[0, 0]

# ============================================================
# 7. mask处理
# ============================================================
mask_pred = (pred_np > threshold).astype(np.uint8) * 255

# 去噪
mask_pred = cv2.medianBlur(mask_pred, 5)

boxes_pred = generate_boxes(mask_pred, min_area=min_area)

print(f"Predicted boxes: {len(boxes_pred)}")

# ============================================================
# 8. 读取GT（用于指标）
# ============================================================
data = np.load(npz_path)
gt_mask = data["labels"][0].astype(np.uint8)

# 尺寸对齐
gt_mask = cv2.resize(gt_mask, (640, 480), interpolation=cv2.INTER_NEAREST)

# ============================================================
# 9. 计算指标
# ============================================================
miou = compute_miou(mask_pred, gt_mask)
dice = compute_dice(mask_pred, gt_mask)

print(f"mIoU: {miou:.4f}")
print(f"Dice: {dice:.4f}")

# ============================================================
# 10. 可视化（单框 + 指标）
# ============================================================
plt.figure(figsize=(8,6))

from matplotlib.colors import LogNorm

vmin = max(np.percentile(temp_resized, 1), 1e-6)
vmax = np.percentile(temp_resized, 55)

plt.imshow(
    temp_resized,
    cmap="inferno",
    norm=LogNorm(vmin=vmin, vmax=vmax)
)

# 只画预测框
for (x, y, w, h) in boxes_pred:
    plt.gca().add_patch(
        plt.Rectangle((x, y), w, h,
                      fill=False, edgecolor='cyan', linewidth=2)
    )

plt.title(f"Predicted defect region\nmIoU={miou:.3f}, Dice={dice:.3f}")
plt.axis("off")

save_path = os.path.join(save_dir, "pred_miou.png")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Saved to: {save_path}")