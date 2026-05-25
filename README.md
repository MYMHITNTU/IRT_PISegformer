# PI-SegFormer-IRT

This repository contains the official implementation for the paper:

**“Thermal response analysis and physics-informed deep learning for active and passive infrared thermography-based delamination detection”**

The project focuses on subsurface delamination detection in reinforced concrete structures using active/passive infrared thermography (IRT), physics-informed deep learning, explainable AI, and thermal response analysis.

---

# Project Structure

```text
├── Dataset/                # Example npz dataset (only one sample is provided due to file size limitations)
├── Dataset_making/         # Thermal data processing and dataset generation
├── Model/                  # PI-SegFormer architecture
├── Prediction/             # Prediction and Grad-CAM visualization results
├── Weights/                # Pretrained model weights
├── Requirements.txt        # Required Python packages
└── README.md
```
---

# Instructions for Use

## 1. Download the code

Clone this repository:

```bash
git clone https://github.com/MYMHITNTU/IRT_PISegformer.git
```

Navigate to the project directory:

```bash
cd IRT_PISegformer
```

---

## 2. Install dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## 3. Download dataset and pretrained weights

For researchers interested in further studies based on this work, the complete dataset and pretrained weights have been uploaded to Zenodo. Please contact me via email to obtain access permission, or wait until this work is officially published.

Dataset download link:

```text
[Add your dataset download link here]
```

Place the downloaded dataset into the `Dataset/` folder and the pretrained weights into the `Weights/` folder.

---

## 4. Run the model

Run the training script:

```bash
python PI_SegFormer.py
```

This will start the training process.

---

## 5. Prediction and visualization

Run the prediction script:

```bash
python predict.py
```

Run the Grad-CAM visualization script:

```bash
python gradcam.py
```

This will generate prediction results and visualize the region of interest (ROI) learned by the model.

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
