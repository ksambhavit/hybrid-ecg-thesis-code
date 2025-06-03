Hybrid ECG Arrhythmia & Stroke-Risk Prediction
This repository contains code for preprocessing, training, and testing three ECG-based models on the PTB-XL dataset:

Pure GNN (MixHop) Baseline

Pure Transformer Baseline

Hybrid Transformer–GNN Model

Follow the instructions below to download PTB-XL, preprocess the data, and evaluate any of these models on arrhythmia classification (multi-label) or continuous stroke-risk regression.

Hardware:

All models were trained/evaluated on NVIDIA GPUs (e.g., A100, RTX 4070).

If running on CPU-only, expect significantly slower training/inference.


Dataset Download
----------------

All experiments use the PTB-XL dataset (v1.0.3). You can obtain it via one of these methods:

Download ZIP (Browser)

Visit:

ruby
Copy
Edit
https://physionet.org/content/ptb-xl/1.0.3/
Click “Download” → “ZIP” → Save the 1.7 GB archive.

Unzip into a local folder, e.g.:

bash
Copy
Edit
unzip ptb-xl-1.0.3.zip -d ./ptbxl_raw
Via wget (Command Line)
Open a terminal and run:

bash
Copy
Edit
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
This mirrors all files under physionet.org/files/ptb-xl/1.0.3/ into your working directory.

AWS S3 (Command Line)
If you have the AWS CLI installed:

bash
Copy
Edit
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ ./ptbxl_raw/
This downloads every file into ./ptbxl_raw/ via S3 public endpoint.

After downloading, ensure you have:

ptbxl_database.csv

scp_statements.csv

The WFDB .dat and .hea files in the records/ subfolders.

Environment Setup
Clone this Repository

bash
Copy
Edit
git clone https://github.com/ksambhavit/hybrid-ecg-thesis-code.git
cd hybrid-ecg-thesis-code
Create & Activate a Virtual Environment
(Recommended: Conda or venv.) For example, with Conda:

bash
Copy
Edit
conda create --name ecg_hybrid python=3.9
conda activate ecg_hybrid
Install Dependencies

bash
Copy
Edit
pip install numpy pandas wfdb scikit-learn tqdm matplotlib
pip install torch torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster torch-geometric
Ensure the installed PyTorch version matches your CUDA (or use CPU-only).

For PyTorch Geometric, follow the official instructions:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Data Preprocessing
Run the preprocessing script to convert raw PTB-XL files into NumPy arrays:

bash
Copy
Edit
python preprocess.py \
  --data_path /path/to/ptbxl_raw \
  --output_dir ./Preprocessed \
  --sampling_rate 500 \
  --compute_global_norm True
--data_path: Folder containing PTB-XL raw files (where ptbxl_database.csv is located).

--output_dir: Destination for the processed .npy files.

--sampling_rate: Should be 500 for PTB-XL.

--compute_global_norm: If True, the script samples 2 000 records to compute global mean/std; otherwise, it does per-batch normalization.

After completion, you’ll see:

bash
Copy
Edit
Preprocessed/
├── X_train.npy      # [N_train × 5000 × 12]
├── X_val.npy        # [N_val × 5000 × 12]
├── X_test.npy       # [N_test × 5000 × 12]
├── y_train.npy      # [N_train × 5]
├── y_val.npy        # [N_val × 5]
├── y_test.npy       # [N_test × 5]
├── meta_train.npy   # [N_train × 2]
├── meta_val.npy     # [N_val × 2]
└── meta_test.npy    # [N_test × 2]
You can now evaluate any model using these preprocessed files.

Testing Pretrained Models
1. Pure GNN Baseline
Location: GNN model/ & checkpoints_better/

Checkpoint
checkpoints_better/better_gnn.pth (1.9 MB) is already included.

Test Script

bash
Copy
Edit
python "GNN model/test_better_gnn.py" \
  --data_dir ./Preprocessed \
  --checkpoint checkpoints_better/better_gnn.pth \
  --batch_size 32
--data_dir: Path to Preprocessed/.

--checkpoint: Path to the .pth file.

--batch_size: (optional) default = 32.

This will load the GNN checkpoint, run inference on X_test.npy, and print final precision, recall, and F1 for arrhythmia classification.

2. Pure Transformer Baseline
Location: Transformer_model/ & Preprocessed/

Checkpoints
Under Preprocessed/, you should already have:

Copy
Edit
transformer_fold1_best.pth
transformer_fold2_best.pth
…
transformer_fold8_best.pth
Each is ~10.2 MB. (If missing, run the training script.)

Test Script

bash
Copy
Edit
python "Transformer_model/test_kfold.py" \
  --data_dir ./Preprocessed \
  --checkpoint_dir ./Preprocessed \
  --batch_size 32
--data_dir: Path to the Preprocessed/ folder.

--checkpoint_dir: Directory containing transformer_fold{i}_best.pth.

--batch_size: (optional) default = 32.

This loads each of the 8 fold checkpoints, evaluates on the test set, and prints the mean ± std of test‐set F1 for arrhythmia classification.

3. Hybrid Transformer–GNN Model
Location: Hybrid model/

Checkpoint
Hybrid model/hyb_try_model_ld.pth (964 KB) is already included.

Hybrid Test Script
A helper script test_hybrid.py (in the repo root) evaluates arrhythmia classification. Example:

bash
Copy
Edit
python test_hybrid.py \
  --data_dir ./Preprocessed \
  --checkpoint "Hybrid model/hyb_try_model_ld.pth" \
  --batch_size 32
--data_dir: Path to Preprocessed/.

--checkpoint: Path to hyb_try_model_ld.pth.

--batch_size: (optional) default = 32.

This will load the hybrid checkpoint, run inference on X_test.npy, and print precision, recall, and F1 for arrhythmia.

Stroke-Risk Regression Test

bash
Copy
Edit
python "Hybrid model/run_finetune_ld.py" \
  --data_dir ./Preprocessed \
  --checkpoint "Hybrid model/hyb_try_model_ld.pth" \
  --batch_size 32 \
  --task regression
--task regression triggers continuous stroke-risk prediction instead of classification.

The script reports final MSE and MAE on the test set.

Directory Structure
Below is the full file/folder hierarchy for reference:

graphql
Copy
Edit
hybrid-ecg-thesis-code/
│
├── preprocess.py
│   └─ (Creates Preprocessed/*.npy)
│
├── Preprocessed/
│   ├── X_train.npy, X_val.npy, X_test.npy            #
│   ├── y_train.npy, y_val.npy, y_test.npy            # All this will be there once you  are done with 
│   ├── meta_train.npy, meta_val.npy, meta_test.npy   # the preprocessing
│   ├── classes.npy                                   #
│   ├── transformer_fold1_best.pth
│   ├── transformer_fold2_best.pth
│   ├── transformer_fold3_best.pth
│   ├── transformer_fold4_best.pth
│   ├── transformer_fold5_best.pth
│   ├── transformer_fold6_best.pth
│   ├── transformer_fold7_best.pth
│   └── transformer_fold8_best.pth
│
├── checkpoints_better/
│   └── better_gnn.pth                   # Pretrained pure GNN
│
├── GNN model/
│   ├── conv_rgnn_dataset.py             # GNN dataset class
│   ├── better_spatial_gnn.py            # MixHop‐based GNN model
│   ├── train_better_gnn.py              # GNN training script
│   └── test_better_gnn.py               # Pure GNN test script
│
├── Transformer_model/
│   ├── transformer_model_new.py         # ECGTransformer definition
│   ├── transformer_train_kfolds.py      # Pure Transformer 8-fold CV
│   └── test_kfold.py                    # Transformer test script
│
├── Hybrid model/
│   ├── dataset_leaddrop.py              # Augmented ECG dataset (lead-dropout)
│   ├── models.py                        # ECGGNNTransformer (hybrid model)
│   ├── train_leaddrop.py                # Hybrid training (classification)
│   ├── run_finetune_ld.py               # Fine-tuning/regression script
│   ├── train.py                         # Utility functions for hybrid training
│   └── hyb_try_model_ld.pth             # Pretrained Hybrid model
│
├── test_hybrid.py                       # Entry-point for hybrid testing
└── README.md                            # ← This file
Citation
If you use or extend this code, please cite:

Kumar Sambhavit
Hybrid Transformer–GNN Architecture for 12-Lead ECG Arrhythmia & Stroke-Risk Prediction.
BSc Thesis, Department of Advanced Computing Sciences, Maastricht University, 2025.

Thank you for using this repository! If you encounter any issues or have questions, feel free to open an issue on GitHub:
https://github.com/ksambhavit/hybrid-ecg-thesis-code