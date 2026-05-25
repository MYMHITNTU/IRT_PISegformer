import os
import json
import numpy as np
import cv2
import pandas as pd

# =========================
# 路径配置
# =========================
excel_folder  = r"..\Dataset_making\Exceldata"
json_folder   = r"..\Dataset_making\Jsondata"
output_folder = r"..\Dataset_making\NPZdata"
conds_file    = r"..\Dataset_making\temp_qt.xlsx"

target_size = (640, 480)  # (width, height)

os.makedirs(output_folder, exist_ok=True)

# =========================
# 读取条件表（q, ΔT, wet）
# =========================
conds_df = pd.read_excel(conds_file, header=None)
conds_dict = {}

for _, row in conds_df.iterrows():
    file_name = str(row[0]).strip()
    delta_T_val = float(row[1])
    q_val = float(row[2])
    wet_val = float(row[3])

    base_name = os.path.splitext(file_name)[0]
    conds_dict[base_name] = (q_val, delta_T_val, wet_val)

# =========================
# 批量处理
# =========================
for filename in os.listdir(excel_folder):
    if not filename.lower().endswith((".xlsx", ".xls")):
        continue

    excel_path = os.path.join(excel_folder, filename)
    base_name = os.path.splitext(filename)[0]
    json_path = os.path.join(json_folder, f"{base_name}.json")
    output_file = os.path.join(output_folder, f"{base_name}.npz")

    # ---- 条件 ----
    if base_name in conds_dict:
        q_val, delta_T_val, wet_val = conds_dict[base_name]
    else:
        print(f"[WARN] {base_name} 未在条件表中找到，使用默认值 q=397, ΔT=30, wet=0")
        q_val, delta_T_val, wet_val = 397.0, 30.0, 0

    if not os.path.exists(json_path):
        print(f"[WARN] {json_path} 不存在，跳过")
        continue

    try:
        # =========================
        # 读取温度 Excel
        # =========================
        df = pd.read_excel(excel_path, header=None)
        temp_map = df.values.astype(np.float32)

        temp_map_resized = cv2.resize(temp_map, target_size)

        # 归一化
        temp_min = temp_map_resized.min()
        temp_max = temp_map_resized.max()
        temp_map_norm = (temp_map_resized - temp_min) / (temp_max - temp_min + 1e-8)

        temps_array = np.expand_dims(temp_map_norm, axis=0)  # (1, H, W)

        # =========================
        # 读取 LabelMe JSON
        # =========================
        with open(json_path, 'r') as f:
            data = json.load(f)

        labels_resized = np.zeros((target_size[1], target_size[0]), dtype=np.float32)

        orig_w = data.get('imageWidth', target_size[0])
        orig_h = data.get('imageHeight', target_size[1])

        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h

        for shape in data.get('shapes', []):
            if shape.get('shape_type') != 'rectangle':
                continue

            points = np.array(shape['points'], dtype=np.float32)
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            points = points.astype(int)

            x1, y1 = points[0]
            x2, y2 = points[1]

            cv2.rectangle(
                labels_resized,
                (x1, y1),
                (x2, y2),
                color=1.0,
                thickness=-1
            )

        labels_array = np.expand_dims(labels_resized, axis=0)  # (1, H, W)

        # =========================
        # 条件向量 [q, ΔT, wet]
        # =========================
        conds_array = np.array([[q_val, delta_T_val, wet_val]], dtype=np.float32)

        # =========================
        # 保存 NPZ
        # =========================
        np.savez(
            output_file,
            temps=temps_array,
            labels=labels_array,
            conds=conds_array
        )

        print(
            f"[OK] {base_name}.npz | "
            f"temps={temps_array.shape}, "
            f"labels={labels_array.shape}, "
            f"conds={conds_array}"
        )

    except Exception as e:
        print(f"[ERROR] 处理 {base_name} 出错: {e}")
