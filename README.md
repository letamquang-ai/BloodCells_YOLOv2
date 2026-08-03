# Blood Cells Detection (Tiny-YOLOv2)

This project trains a Tiny-YOLOv2 model (via `darkflow`) to detect blood cell types (e.g. RBC, WBC, Platelets) from microscope images.

## Project Structure

```
.
├── cfg/
│   └── tiny-yolo-voc-3c.cfg
├── bin/
│   └── tiny-yolo-voc.weights
├── weights/
├── dataset/
│   ├── Training/
│   │   ├── Images/
│   │   └── Annotations/
│   └── Validation/
│       ├── Images/
│       └── Annotations/
├── ckpt/
├── losses.txt
├── pre_losses.txt
└── README.md
```

## 1. Environment Setup

Create and activate a dedicated conda environment with Python 3.7:

```bash
conda create -n bloodcells python=3.7
conda activate bloodcells
```

Install TensorFlow GPU 2.2.0 via conda:

```bash
conda install tensorflow-gpu==2.2.0
```

Install the remaining Python dependencies via pip:

```bash
pip install tf-slim==1.1.0 cython==0.29.21 opencv-python==4.2.0.32
```

## 2. Build the Cython Extensions

`darkflow` requires its Cython extensions to be built in place before use:

```bash
python setup.py build_ext --inplace
```

## Requirements Summary

| Package          | Version   |
|-------------------|-----------|
| Python             | 3.7       |
| tensorflow-gpu      | 2.2.0     |
| tf-slim             | 1.1.0     |
| cython              | 0.29.21   |
| opencv-python       | 4.2.0.32  |
