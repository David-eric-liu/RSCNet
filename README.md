# RSCNet: Visible–Infrared Person Re-Identification

**PyTorch implementation of RSCNet for visible–infrared person re-identification (VI-ReID).**

RSCNet studies cross-modality feature learning between RGB (visible) and infrared (IR) pedestrian images. This repository contains training and evaluation scripts for **SYSU-MM01** and **RegDB**, with a ResNet-50-based feature extractor, an RCSRI-v2 interaction module, part-based embeddings, and a hybrid cross-center objective.

> **Release status:** Research-code snapshot. The code is provided for inspection and reproduction, but this archive does **not** include datasets, pretrained RSCNet checkpoints, benchmark logs, or an automated test suite. Results have not been independently reproduced as part of preparing this README.

## 1. Highlights

- **Cross-modality feature learning:** a shared ResNet backbone followed by modality-specific high-level feature branches for visible and infrared inputs.
- **RCSRI-v2 (Relation–Channel Selective Residual Interaction):** inserted between `layer3` and `layer4` during paired RGB–IR training. It combines spatial/relation gating, channel gating, a learnable bounded residual scale, and scale protection.
- **Part-based representation:** spatially pooled features are passed to multiple classifiers to learn discriminative local representations.
- **Hybrid cross-center loss:** encourages samples to approach the corresponding cross-modality identity centers and includes an auxiliary center-distance ratio term.
- **Benchmark protocols:** supports SYSU-MM01 all-search/indoor-search evaluation and RegDB visible-to-infrared or infrared-to-visible retrieval.

**Important implementation detail:** in the current `model.py`, the RCSRI-v2 module is applied only for paired inputs **during training** (`modal == 0` and `self.training`). Single-modality inference bypasses the interaction module. The code is the source of truth if a description of the method elsewhere differs from this snapshot.

## 2. Repository Structure

```text
RSCNet/
├── README.md                    # Project documentation
├── .gitignore                   # Excludes outputs, datasets, and checkpoints
├── train.py                     # Training and per-epoch evaluation
├── test.py                      # Standalone benchmark evaluation
├── model.py                     # Shared backbone, modality branches, embeddings
├── rcsri_v2.py                  # RCSRI-v2 interaction module
├── hybrid_cross_center_loss.py  # Hybrid cross-center objective
├── cross_center_loss.py         # Additional cross-center loss implementation
├── loss.py                      # Triplet-loss and related utilities
├── resnet.py                    # ResNet backbone and ImageNet initialization
├── data_loader.py               # SYSU-MM01 / RegDB dataset loaders
├── data_manager.py              # Query/gallery generation and test splits
├── pre_process_sysu.py          # Generate SYSU-MM01 training NumPy arrays
├── eval_metrics.py              # CMC, mAP, and mINP evaluation
├── random_erasing.py            # Data augmentation utilities
├── random_linear.py             # Additional data augmentation implementation
└── utils.py                     # Sampling, logging, and helper functions
```

## 3. Environment

A CUDA-capable NVIDIA GPU is recommended and is currently required by the training and evaluation paths that explicitly call `.cuda()`.

The project imports PyTorch, torchvision, NumPy, Pillow, SciPy, scikit-learn, Matplotlib, and (in `random_linear.py`) OpenCV. For example, you can create an environment as follows:

```bash
conda create -n rscnet python=3.10 -y
conda activate rscnet

# Install a PyTorch/torchvision build appropriate for your CUDA driver.
# Example version pairing (not a newly verified compatibility test):
pip install torch==2.1.0 torchvision==0.16.0
pip install "numpy<2" "Pillow<10" scipy scikit-learn matplotlib opencv-python
```

Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) if a specific CUDA wheel is needed. Older Pillow compatibility is relevant because the source code uses `Image.ANTIALIAS`, which was removed in Pillow 10. The ResNet implementation requests pretrained ImageNet weights at startup, so an initial download or an available cache may be required.

## 4. Dataset Preparation

**Datasets are not included.** Obtain the official datasets according to their respective access terms. The scripts currently use **hard-coded absolute paths**; update them to match your machine before running anything:

| File | Current dataset path |
| --- | --- |
| `train.py` | `/root/autodl-tmp/datasets/SYSU-MM01/` or `/root/autodl-tmp/datasets/RegDB/` |
| `test.py` | `/root/autodl-tmp/datasets/SYSU-MM01/` or `/root/autodl-tmp/datasets/RegDB/` |
| `pre_process_sysu.py` | `/root/autodl-tmp/datasets/SYSU-MM01/` |

### SYSU-MM01

Place the extracted benchmark under your configured SYSU-MM01 directory. The scripts expect the `cam1/`–`cam6/` camera folders and the official `exp/` split files, including `train_id.txt`, `val_id.txt`, and `test_id.txt`.

Before training, run:

```bash
python pre_process_sysu.py
```

This generates four files in the configured SYSU-MM01 directory:

```text
train_rgb_resized_img.npy
train_rgb_resized_label.npy
train_ir_resized_img.npy
train_ir_resized_label.npy
```

### RegDB

Place the official image folders and `idx/` trial-split files in the configured RegDB directory. The loader expects files such as:

```text
idx/train_visible_1.txt
idx/train_thermal_1.txt
idx/test_visible_1.txt
idx/test_thermal_1.txt
```

The index files should point to images relative to the RegDB root and contain a person label on each line. Training/evaluation splits are trial-dependent.

## 5. Training

Run commands **from the repository root** after setting the dataset paths and installing the dependencies.

### SYSU-MM01

```bash
python train.py \
  --dataset sysu \
  --gpu 0 \
  --mode all \
  --arch resnet50 \
  --img_h 384 --img_w 192 \
  --batch-size 7 --num_pos 8 \
  --part_num 12 \
  --lr 0.03 --optim sgd \
  --margin 0.3 --margin_cc 0.1 \
  --dist-type l2 \
  --w_hc 3.0 \
  --workers 8
```

For indoor-search evaluation during training, use `--mode indoor`.

### RegDB

```bash
python train.py \
  --dataset regdb \
  --trial 1 \
  --gpu 0 \
  --batch-size 7 --num_pos 8 \
  --part_num 12 \
  --lr 0.03 --optim sgd \
  --w_hc 3.0
```

Train separate checkpoints for different RegDB trial splits when following a multi-trial evaluation protocol.

**Training behavior in this snapshot:**

- `train.py` trains from epoch 0 through 100 by default; a resumed run starts from the stored checkpoint epoch.
- The loss combines identity classification, triplet loss, hybrid cross-center loss, and a global classification term.
- `--w_hc` weights the hybrid cross-center objective; `HybridCrossCenterLoss` also contains an internal epoch-dependent coefficient (`beta_max=0.02`, `warm_epoch=30`) for its auxiliary term. Do not equate this implementation automatically with a different paper loss equation.
- Checkpoints are written to `save_model/` by default, including the best Rank-1 and best mAP snapshots. Logs are written to `log/` and `log/vis_log/`.
- Keep `--part_num 12` for the unmodified `train.py`: its evaluation buffers currently assume `12 × 512` dimensions.
- The `--test-only` flag is declared but does not skip training in `train.py`; use `test.py` for standalone evaluation.

## 6. Evaluation

The standalone evaluation script reports **CMC (Rank-1/5/10/20), mAP, and mINP**. Supply an **existing** checkpoint filename with `--resume` (relative to `--model_path`). No RSCNet checkpoint is included in this archive.

### SYSU-MM01: all-search

```bash
python test.py \
  --dataset sysu \
  --gpu 0 \
  --mode all \
  --part_num 12 \
  --model_path save_model/ \
  --resume YOUR_SYSU_CHECKPOINT.t
```

Use `--mode indoor` for indoor-search. The script evaluates 10 sampled gallery trials and prints average metrics. It also writes `pytorch_result.mat` during SYSU evaluation.

### RegDB: important checks before evaluation

**Do not assume the current `test.py` works unchanged with RegDB checkpoints.** In the `regdb` branch, `test.py` sets `n_class = 395`, although the standard RegDB training split has 206 training identities. Match `n_class` to the checkpoint architecture (usually `206` for RegDB) before loading its state dictionary.

Also, `test.py` loops over trials 1–10 but reloads the **same** `--resume` checkpoint in every iteration. For a correct separately trained 10-trial benchmark, adapt the script to load the matching trial-specific checkpoint in each iteration, or evaluate each trial explicitly. Do not interpret the default loop as a valid multi-trial result without that adjustment.

After making those checks, the relevant options are `--dataset regdb`, `--resume YOUR_REGDB_CHECKPOINT.t`, and `--tvsearch` for the reverse thermal-to-visible retrieval direction. Without `--tvsearch`, the script uses visible-to-thermal retrieval.

## 7. Metrics

- **Rank-k (CMC):** fraction of queries for which the correct identity appears within the top-k gallery matches.
- **mAP:** mean average precision over the query set.
- **mINP:** mean inverse negative penalty, reflecting the retrieval of harder relevant matches.

The code implements the benchmark-specific filtering rules in `eval_metrics.py`. For final paper comparisons, follow the official protocols and report the evaluation direction, search mode, and trial aggregation clearly.

## 8. Reproducibility Notes

- Fix dataset paths in the three files listed above; there is currently no `--data_path` command-line option.
- Use the correct checkpoint for each dataset and RegDB trial. Checkpoint filenames depend on the training arguments, including `--train_mode`, `--w_hc`, and `--trial`.
- The training and test scripts do not have identical defaults; **always specify** the checkpoint name when testing.
- The code uses a GPU-oriented implementation and includes fixed dimensions and paths. It has not been converted into a platform-independent package.
- This README documents the **uploaded code snapshot**, not a claim that every experiment or final paper formulation has been implemented identically.

## 9. Citation and License

If you use this research code, please acknowledge the corresponding project or paper once its official citation details are available. A verified BibTeX entry and publication link have **not** been added here to avoid inventing bibliographic information.

**License:** No license file is included in this snapshot. Before redistributing or reusing the code, confirm the rights and attribution requirements for all original and adapted components and add an appropriate license if you are authorized to do so.
