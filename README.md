# PI-SegFormer-IRT

This repository contains the official implementation for the paper:

**“Thermal response analysis and physics-informed deep learning for active and passive infrared thermography-based delamination detection”**

The project focuses on subsurface delamination detection in reinforced concrete structures using active/passive infrared thermography (IRT), physics-informed deep learning, explainable AI, and numerical simulation.

---

# Features

- Active and passive infrared thermography (IRT) analysis
- Physics-Informed SegFormer (PI-SegFormer)
- Delamination localization and segmentation
- Dry/wet condition analysis
- Thermal response evolution analysis
- Grad-CAM visualization
- Numerical simulation-assisted thermal analysis

---

# Project Structure

```
│
├── Dataset/                # Thermal image datasets
├── Weights/                # Pretrained model weights
├── Model/                  # PI-SegFormer architecture
├── Prediction/             # Prediction and visualization results
├── Simulation/             # Numerical simulation results
├── Utils/                  # Utility functions
├── software.py             # Main software interface
└── README.md
Instructions for Use
1. Download the code

Clone this repository:

git clone https://github.com/YourGitHubName/PI-SegFormer-IRT.git
2. Download pretrained weights

Download the pretrained weights and place them into the Weights/ folder.

Weight download link:

[Add your download link here]
3. Run the software

Navigate to the project directory:

cd PI-SegFormer-IRT

Run:

python software.py

This will open the software interface for IRT defect detection and visualization.

Dataset

The dataset includes:

Active IRT thermal images
Passive IRT thermal images
Dry/wet condition datasets
Delamination annotations
Temperature matrix data

For access to the complete dataset, please contact:

[Your Email]

Example datasets and sample files are provided for testing and training.

Environment Configuration

The code was tested under:

Python 3.9
CUDA 11.6

Required packages:

torch==1.12.1
torchvision==0.13.1
mmcv==1.6.2
timm==0.4.12
opencv-python
numpy
matplotlib
scikit-image
pandas
Pillow
tkinter

Install dependencies using:

pip install -r requirements.txt
Citation

If you find this work useful, please cite:

[Your paper citation here]
Grad-CAM Visualization

The repository includes Grad-CAM visualization tools for interpreting defect-related thermal attention regions in the PI-SegFormer model.

Numerical Simulation

COMSOL-based numerical simulations were conducted to investigate the three-dimensional thermal diffusion behavior associated with subsurface delamination defects under both active and passive heating conditions.

License

This project is released under the MIT License.

Contact

Yiming Ma
Nanyang Technological University (NTU)
Email: [Your Email]
