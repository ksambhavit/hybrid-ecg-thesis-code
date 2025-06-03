# Hybrid ECG Arrhythmia & Stroke-Risk Prediction
## This repository contains code for preprocessing, training, and testing three ECG-based models on the PTB-XL dataset:

Pure GNN (MixHop) Baseline

Pure Transformer Baseline

Hybrid Transformer–GNN Model

Follow the instructions below to download PTB-XL, preprocess the data, and evaluate any of these models on arrhythmia classification (multi-label) or continuous stroke-risk regression.

Hardware:

All models were trained/evaluated on NVIDIA GPUs (e.g., A100, RTX 4070).

If running on CPU-only, expect significantly slower training/inference.


Paste your rich text content here. You can paste directly from Word or other rich text sources.

## Dataset Download

All experiments use the PTB-XL dataset (v1.0.3). You can obtain it via one of these methods:

1. 1.  **Download ZIP (Browser)**
  Visit:
     
1.     1.     `https://physionet.org/content/ptb-xl/1.0.3/`
  Click “Download” → “ZIP” → Save the 1.7 GB archive.
    
  Unzip into a local folder, e.g.:
    
1.     1.     `unzip ptb-xl-1.0.3.zip -d ./ptbxl_raw`
    
  **Via `wget` (Command Line)**  
     Open a terminal and run:
  
1.     `wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/`
     
    This mirrors all files under `physionet.org/files/ptb-xl/1.0.3/` into your working directory.
    
1. 3.  **AWS S3 (Command Line)**  
     If you have the AWS CLI installed:
   
1.     `aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ ./ptbxl_raw/`
  
    This downloads every file into `./ptbxl_raw/` via S3 public endpoint.
     

After downloading, ensure you have:

* *   `ptbxl_database.csv`
*     
* *   `scp_statements.csv`
*     
* *   The WFDB `.dat` and `.hea` files in the `records/` subfolders.
*     

* * *

## Environment Setup

  **Clone this Repository**

      
1.     `git clone https://github.com/ksambhavit/hybrid-ecg-thesis-code.git cd hybrid-ecg-thesis-code`
 
 **Create & Activate a Virtual Environment**  
   (Recommended: Conda or `venv`.) For example, with Conda:
       
1.     `conda create --name ecg_hybrid python=3.9 conda activate ecg_hybrid`
     
 **Install Dependencies**
        
1.     `pip install numpy pandas wfdb scikit-learn tqdm matplotlib pip install torch torchvision torchaudio pip install torch-scatter torch-sparse torch-cluster torch-geometric`
   
     * *   Ensure the installed PyTorch version matches your CUDA (or use CPU-only).
     *     
    * *   For PyTorch Geometric, follow the official instructions:  
 1.    *     [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
     *     

* * *

## Data Preprocessing

Run the preprocessing script to convert raw PTB-XL files into NumPy arrays:

bash

`python preprocess.py \   --data_path /path/to/ptbxl_raw \   --output_dir ./Preprocessed \   --sampling_rate 500 \   --compute_global_norm True`

* *   **`--data_path`**: Folder containing PTB-XL raw files (where `ptbxl_database.csv` is located).
*     
* *   **`--output_dir`**: Destination for the processed `.npy` files.
*     
* *   **`--sampling_rate`**: Should be `500` for PTB-XL.
*     
* *   **`--compute_global_norm`**: If `True`, the script samples 2 000 records to compute global mean/std; otherwise, it does per-batch normalization.
*     

After completion, you’ll see:


- **Preprocessed/**
  - `X_train.npy` &nbsp;&nbsp;&nbsp;# [N_train × 5000 × 12]
  - `X_val.npy` &nbsp;&nbsp;&nbsp;# [N_val × 5000 × 12]
  - `X_test.npy` &nbsp;&nbsp;&nbsp;# [N_test × 5000 × 12]
  - `y_train.npy` &nbsp;&nbsp;&nbsp;# [N_train × 5]
  - `y_val.npy` &nbsp;&nbsp;&nbsp;# [N_val × 5]
  - `y_test.npy` &nbsp;&nbsp;&nbsp;# [N_test × 5]
  - `meta_train.npy` &nbsp;&nbsp;&nbsp;# [N_train × 2]
  - `meta_val.npy` &nbsp;&nbsp;&nbsp;# [N_val × 2]
  - `meta_test.npy` &nbsp;&nbsp;&nbsp;# [N_test × 2]

You can now evaluate any model using these preprocessed files.

* * *

## Testing Pretrained Models

### 1\. Pure GNN Baseline

* *   **Location**: `GNN model/` & `checkpoints_better/`
*     1  **Checkpoint**  
    `checkpoints_better/better_gnn.pth` (1.9 MB) is already included.
       
     **Test Script**
     

    
     `python "GNN model/test_better_gnn.py" \   --data_dir ./Preprocessed \   --checkpoint checkpoints_better/better_gnn.pth \   --batch_size 32`
       
    * *   **`--data_dir`**: Path to `Preprocessed/`.
    *     
    * *   **`--checkpoint`**: Path to the `.pth` file.
    *     
   * *   **`--batch_size`**: (optional) default = 32.
    *     

This will load the GNN checkpoint, run inference on `X_test.npy`, and print final precision, recall, and F1 for arrhythmia classification.

* * *

### 2\. Pure Transformer Baseline

* *   **Location**: `Transformer_model/` & `Preprocessed/`
*      **Checkpoints**  
     Under `Preprocessed/`, you should already have:
       
         
       
1.     `transformer_fold1_best.pth transformer_fold2_best.pth … transformer_fold8_best.pth`
       
     Each is ~10.2 MB. (If missing, run the training script.)     
  **Test Script**     
1.     `python "Transformer_model/test_kfold.py" \   --data_dir ./Preprocessed \   --checkpoint_dir ./Preprocessed \   --batch_size 32`
       
    * *   **`--data_dir`**: Path to the `Preprocessed/` folder.
     *     
    * *   **`--checkpoint_dir`**: Directory containing `transformer_fold{i}_best.pth`.
    *     
    * *   **`--batch_size`**: (optional) default = 32.
    *     

This loads each of the 8 fold checkpoints, evaluates on the test set, and prints the mean ± std of test‐set F1 for arrhythmia classification.

* * *

### 3\. Hybrid Transformer–GNN Model

* *   **Location**: `Hybrid model/`
*       **Checkpoint**  
    `Hybrid model/hyb_try_model_ld.pth` (964 KB) is already included.
       
   .  **Hybrid Test Script**  
       A helper script `test_hybrid.py` (in the repo root) evaluates arrhythmia classification. Example:
         
     
1.     `python test_hybrid.py \   --data_dir ./Preprocessed \   --checkpoint "Hybrid model/hyb_try_model_ld.pth" \   --batch_size 32`
     
     * *   **`--data_dir`**: Path to `Preprocessed/`.
     *     
     * *   **`--checkpoint`**: Path to `hyb_try_model_ld.pth`.
       *
        * *   **`--batch_size`**: (optional) default = 32.
     *     

This will load the hybrid checkpoint, run inference on `X_test.npy`, and print precision, recall, and F1 for arrhythmia.

1. 3.  **Stroke-Risk Regression Test**
       
1.     `python "Hybrid model/run_finetune_ld.py" \   --data_dir ./Preprocessed \   --checkpoint "Hybrid model/hyb_try_model_ld.pth" \   --batch_size 32 \   --task regression`
   
     * *   **`--task regression`** triggers continuous stroke-risk prediction instead of classification.
     *     
      * *   The script reports final MSE and MAE on the test set.
      *     

* * *

## Directory Structure

Below is the full file/folder hierarchy for reference:

# Directory Structure

- **hybrid-ecg-thesis-code/**  
  - `preprocess.py`  
    - (Converts PTB-XL raw files → `Preprocessed/*.npy`)  
  - **Preprocessed/**  
    - `X_train.npy`  
    - `X_val.npy`  
    - `X_test.npy`  
    - `y_train.npy`  
    - `y_val.npy`  
    - `y_test.npy`  
    - `meta_train.npy`  
    - `meta_val.npy`  
    - `meta_test.npy`  
    - `classes.npy`  
    - `transformer_fold1_best.pth`  
    - `transformer_fold2_best.pth`  
    - `transformer_fold3_best.pth`  
    - `transformer_fold4_best.pth`  
    - `transformer_fold5_best.pth`  
    - `transformer_fold6_best.pth`  
    - `transformer_fold7_best.pth`  
    - `transformer_fold8_best.pth`  
  - **checkpoints_better/**  
    - `better_gnn.pth`  
  - **GNN model/**  
    - `conv_rgnn_dataset.py`  
    - `better_spatial_gnn.py`  
    - `train_better_gnn.py`  
    - `test_better_gnn.py`  
  - **Transformer_model/**  
    - `transformer_model_new.py`  
    - `transformer_train_kfolds.py`  
    - `test_kfold.py`  
  - **Hybrid model/**  
    - `dataset_leaddrop.py`  
    - `models.py`  
    - `train_leaddrop.py`  
    - `run_finetune_ld.py`  
    - `train.py`  
    - `load_ckpt.py`  
    - `test_hybrid.py`  
    - `hyb_try_model_ld.pth`  
  - `test_hybrid.py`  
  - `README.md`  


## Citation

If you use or extend this code, please cite:

> **Kumar Sambhavit**  
> _Hybrid Transformer–GNN Architecture for 12-Lead ECG Arrhythmia & Stroke-Risk Prediction._  
> BSc Thesis, Department of Advanced Computing Sciences, Maastricht University, 2025.

* * *

Thank you for using this repository! If you encounter any issues or have questions, feel free to open an issue on GitHub:  
[https://github.com/ksambhavit/hybrid-ecg-thesis-code](https://github.com/ksambhavit/hybrid-ecg-thesis-code)
