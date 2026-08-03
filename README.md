# Blood Cells Detection (Tiny-YOLO-VOC)

This project trains a Tiny-YOLO-VOC model (via `darkflow`) to detect blood cell types (e.g. RBC, WBC, Platelets) from microscope images.

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

## 3. Training

### Stage 1 — Initial Training

Train from the pretrained `tiny-yolo-voc` weights with a learning rate of `1e-5` for 50 epochs:

```bash
python flow --model cfg/tiny-yolo-voc-3c.cfg \
    --load bin/tiny-yolo-voc.weights \
    --train \
    --gpu 1.0 \
    --annotation dataset/Training/Annotations \
    --annotation_valid dataset/Validation/Annotations \
    --dataset dataset/Training/Images \
    --validation dataset/Validation/Images \
    --lr 1e-5 \
    --epoch 50
```

After this stage finishes, back up the loss log so it isn't overwritten in the next stage:

```bash
mv losses.txt pre_losses.txt
```

### Stage 2 — Fine-tuning

Resume training from the best checkpoint of Stage 1, using a much smaller learning rate (`1e-7`) for another 50 epochs:

```bash
python flow --model cfg/tiny-yolo-voc-3c.cfg \
    --load bin/tiny-yolo-voc.weights \
    --train \
    --gpu 1.0 \
    --annotation dataset/Training/Annotations \
    --annotation_valid dataset/Validation/Annotations \
    --dataset dataset/Training/Images \
    --validation dataset/Validation/Images \
    --load best_ckpt \
    --lr 1e-7 \
    --epoch 50
```

> **Note on `--load best_ckpt`:** replace `best_ckpt` with the checkpoint step number that had the lowest validation loss from Stage 1 (found in `pre_losses.txt` or your validation logs), e.g. `--load -2035`.

## 4. Choosing a Confidence Threshold

Before evaluating, decide on a confidence threshold to filter low-confidence detections. This is typically chosen by:

- Reviewing precision/recall at different thresholds on the validation set
- Picking the threshold that gives the best balance of precision and recall (e.g. via an F1 curve)

Pass the chosen value with the `--threshold` flag during evaluation/inference.

## 5. Evaluation

Run evaluation using the fine-tuned checkpoint and selected confidence threshold:

```bash
python flow --model cfg/tiny-yolo-voc-3c.cfg \
    --load best_ckpt \
    --threshold <chosen_threshold> \
    --test dataset/Validation/Images \
    --annotation dataset/Validation/Annotations
```

Review the resulting metrics (precision, recall, mAP) to assess model performance.

## Project Structure

```
.
├── cfg/
│   └── tiny-yolo-voc-3c.cfg
├── bin/
│   └── tiny-yolo-voc.weights
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

## Requirements Summary

| Package          | Version   |
|-------------------|-----------|
| Python             | 3.7       |
| tensorflow-gpu      | 2.2.0     |
| tf-slim             | 1.1.0     |
| cython              | 0.29.21   |
| opencv-python       | 4.2.0.32  |
