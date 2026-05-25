# PI-SegFormer-IRT

This repository contains the official implementation for the paper:

**“Thermal response analysis and physics-informed deep learning for active and passive infrared thermography-based delamination detection”**

The project focuses on subsurface delamination detection in reinforced concrete structures using active/passive infrared thermography (IRT), physics-informed deep learning, explainable AI, and thermal response analysis.

---

# Project Structure

```text
│
├── Dataset/                # Example npz dataset (only one sample is provided due to file size limitations)
├── Dataset_making/         # Thermal data processing and dataset generation
├── Model/                  # PI-SegFormer architecture
├── Prediction/             # Prediction and Grad-CAM visualization results
├── Weights/                # Pretrained model weights
└── README.md
```

---

# Instructions for Use

## 1. Download the code

Clone this repository:

```bash
git clone https://github.com/MYMHITNTU/IRT_PISegformer.git
```

---

## 2. Download dataset

For researchers interested in further studies based on this work, the complete dataset has been uploaded to Baidu Netdisk. Please contact me via email to obtain the access password.

Dataset download link:

```text
[Add your dataset download link here]
```

---

## 3. Run the model

Navigate to the project directory:

```bash
cd IRT_PISegformer
```

Run the training script:

```bash
python PI_SegFormer.py
```

This will start the training process.

---

## 4. Prediction and visualization

Run the prediction script to evaluate the segmentation performance.

Run the Grad-CAM script to visualize the region of interest (ROI) learned by the model.

---

# Environment Configuration

The code was tested under:

```text
Python 3.9
CUDA 11.6
```

Required packages:

```text
torch==1.12.1
torchvision==0.13.1
transformers
opencv-python
numpy
matplotlib
scikit-learn
scikit-image
pandas
json
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# Citation

If you find this work useful, please cite our paper. The citation information will be updated after publication.

---

# License

This project is released under the MIT License.

---

# Contact

Yiming Ma  
Nanyang Technological University (NTU)  
Email: yiming.ma@ntu.edu.sg
