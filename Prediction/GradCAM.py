import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from Model.PI_Segformer import IRTDefectDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 配置
# =====================================================
model_path = r"..\Weights\epoch_200.pth"

npz_folder = r"..\Dataset"
save_dir = r"..\Prediction\GradCAM_results"
os.makedirs(save_dir, exist_ok=True)

threshold = 0.5
min_area = 150

sample_ids = ["TE_00681"]

# =====================================================
# 模型
# =====================================================
model = IRTDefectDetector(in_ch=1, out_ch=128).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =====================================================
# hook
# =====================================================
features = None
grads = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    global grads
    grads = grad_out[0]

model.backbone.fuse.register_forward_hook(forward_hook)
model.backbone.fuse.register_backward_hook(backward_hook)

# =====================================================
# box函数
# =====================================================
def generate_boxes(mask, min_area=150):
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        boxes.append((x, y, w, h))
    return boxes

# =====================================================
# 可视化
# =====================================================
plt.figure(figsize=(12, 3 * len(sample_ids)))

for i, sid in enumerate(sample_ids):

    npz_path = os.path.join(npz_folder, sid + ".npz")

    if not os.path.exists(npz_path):
        print(f"Not found: {npz_path}")
        continue

    data = np.load(npz_path)

    temp = data["temps"].astype(np.float32)
    cond = data["conds"].astype(np.float32)

    # 归一化
    temp = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)

    # reshape
    if temp.ndim == 2:
        temp = temp[None, None, :, :]
    elif temp.ndim == 3:
        temp = temp[:, None, :, :]

    if cond.ndim == 1:
        cond = cond[None, :]

    temp_tensor = torch.tensor(temp, dtype=torch.float32).to(device)
    cond_tensor = torch.tensor(cond, dtype=torch.float32).to(device)

    cond_tensor[:, 0] /= 300.0
    cond_tensor[:, 1] /= 200.0

    # ===== forward =====
    pred = model(temp_tensor, cond_tensor)
    pred_np = pred.detach().cpu().numpy()[0, 0]

    # ===== mask =====
    mask_pred = (pred_np > threshold).astype(np.float32)
    mask_pred = cv2.medianBlur((mask_pred * 255).astype(np.uint8), 5)
    mask_pred = (mask_pred > 0).astype(np.float32)

    # ===== boxes =====
    boxes = generate_boxes(mask_pred, min_area=min_area)

    # ===== Grad-CAM =====
    target = (pred[0, 0] * torch.tensor(mask_pred).to(device)).sum()

    model.zero_grad()
    target.backward()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True)

    cam = torch.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()

    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = cam * mask_pred

    # ===== 原始灰度图 =====
    img = temp_tensor.cpu().numpy()[0, 0]

    # ===== 彩色 thermal =====
    img_color = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # ===== box图 =====
    img_box = img_color.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(img_box, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # ===== Grad-CAM =====
    img_gray = np.stack([img]*3, axis=-1)

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = img_gray.copy()
    overlay[mask_pred > 0] = (
        0.6 * heatmap[mask_pred > 0] / 255 +
        0.4 * img_gray[mask_pred > 0]
    )

    # ===== 标题 =====
    q_val = cond[0, 0]

    # ===== plot =====
    plt.subplot(len(sample_ids), 3, i*3 + 1)
    plt.title(f"{sid} | q={q_val:.1f}")
    plt.imshow(img_color)
    plt.axis("off")

    plt.subplot(len(sample_ids), 3, i*3 + 2)
    plt.title(f"Boxes ({len(boxes)})")
    plt.imshow(img_box)
    plt.axis("off")

    plt.subplot(len(sample_ids), 3, i*3 + 3)
    plt.title("Grad-CAM")
    plt.imshow(overlay)
    plt.axis("off")

# 保存
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "final_mixed_style.png"), dpi=300)
plt.close()

print("Visualization saved!")