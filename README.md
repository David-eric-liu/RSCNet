# RSCNet

PyTorch implementation of **RSCNet** for Visible-Infrared Person Re-Identification (VI-ReID).

## Requirements

- Python 3.10
- PyTorch 2.1.0
- CUDA
- NVIDIA RTX 4090 GPU

## Prepare Datasets

**(1) RegDB:** Download the RegDB dataset and organize the images and split files under the `RegDB/` directory.

**(2) SYSU-MM01:** Download the SYSU-MM01 dataset and run the following command to preprocess the training images:

```bash
python pre_process_sysu.py
```

Before running the code, update the dataset paths in `train.py`, `test.py`, and `pre_process_sysu.py`.

## Training

We use **one NVIDIA RTX 4090 GPU** for training.

**SYSU-MM01:**

```bash
python train.py --gpu 0 --dataset sysu 
```

**RegDB:**

```bash
python train.py --gpu 0 --dataset regdb 
```

## Evaluation

**SYSU-MM01 (All-search):**

```bash
python test.py --gpu 0 --dataset sysu --mode all --resume YOUR_CHECKPOINT.t
```

For indoor-search evaluation:

```bash
python test.py --gpu 0 --dataset sysu --mode indoor --resume YOUR_CHECKPOINT.t
```

Replace `YOUR_CHECKPOINT.t` with the checkpoint filename in `save_model/`.

For RegDB evaluation, ensure that the classifier size and trial-specific checkpoints in `test.py` match the corresponding trained models.
