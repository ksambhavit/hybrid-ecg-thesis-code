Hybrid ECG Arrhythmia & Stroke-Risk Prediction
==============================================

This repository contains code for preprocessing, training, and testing three ECG-based models on the PTB-XL dataset:

*   **Pure GNN (MixHop) Baseline**
    
*   **Pure Transformer Baseline**
    
*   **Hybrid Transformer–GNN Model**
    

Follow the instructions below to download PTB-XL, preprocess the data, and evaluate any of these models on arrhythmia classification (multi-label) or continuous stroke-risk regression.

Table of Contents
-----------------

*   [Dataset Download](#dataset-download)
    
*   [Environment Setup](#environment-setup)
    
*   [Data Preprocessing](#data-preprocessing)
    
*   [Testing Pretrained Models](#testing-pretrained-models)
    
*   [Directory Structure](#directory-structure)
    
*   [Required Python Imports](#required-python-imports)
    
*   [Citation](#citation)
    

Dataset Download
----------------

All experiments use the PTB-XL dataset (v1.0.3). You can obtain it via one of these methods:

### Download via Browser (ZIP)

1.  Visit: [https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/)
    
2.  Click **Download → ZIP** and save the 1.7 GB archive.
    
3.  bashCopyEditunzip ptb-xl-1.0.3.zip -d ./ptbxl\_raw
    
4.  After unzipping, ensure you have:
    
    *   ptbxl\_database.csv
        
    *   scp\_statements.csv
        
    *   The WFDB .dat and .hea files in the records/ subfolders.
        

### Download via wget (Command Line)

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditwget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/   `

This mirrors all files under physionet.org/files/ptb-xl/1.0.3/ into your working directory.

### Download via AWS S3 (Command Line)

If you have the AWS CLI installed:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditaws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ ./ptbxl_raw/   `

This downloads every file into ./ptbxl\_raw/ via the public S3 endpoint.

Environment Setup
-----------------

### Clone This Repository

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditgit clone https://github.com/ksambhavit/hybrid-ecg-thesis-code.git  cd hybrid-ecg-thesis-code   `

### Create & Activate a Virtual Environment

**Conda (Recommended):**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditconda create --name ecg_hybrid python=3.9 -y  conda activate ecg_hybrid   `

**Or venv:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython3.9 -m venv ecg_hybrid  source ecg_hybrid/bin/activate   `

### Install Dependencies

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpip install numpy pandas wfdb scikit-learn tqdm matplotlib  pip install torch torchvision torchaudio  pip install torch-scatter torch-sparse torch-cluster torch-geometric   `

> Ensure the installed PyTorch version matches your CUDA (or use CPU-only).

For PyTorch Geometric, always follow the official instructions for your system:[https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Data Preprocessing
------------------

Convert the raw PTB-XL files into NumPy arrays using the provided script:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython preprocess.py \    --data_path /path/to/ptbxl_raw \    --output_dir ./Preprocessed \    --sampling_rate 500 \    --compute_global_norm True   `

*   \--data\_path: Folder with raw PTB-XL files (where ptbxl\_database.csv is located).
    
*   \--output\_dir: Output folder for .npy files.
    
*   \--sampling\_rate: Should be 500 for PTB-XL.
    
*   \--compute\_global\_norm: If True, uses 2,000 random records to compute global normalization stats.
    

After completion, you’ll see:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   textCopyEditPreprocessed/  ├── X_train.npy      # [N_train × 5000 × 12]  ├── X_val.npy        # [N_val × 5000 × 12]  ├── X_test.npy       # [N_test × 5000 × 12]  ├── y_train.npy      # [N_train × 5]  ├── y_val.npy        # [N_val × 5]  ├── y_test.npy       # [N_test × 5]  ├── meta_train.npy   # [N_train × 2]  ├── meta_val.npy     # [N_val × 2]  └── meta_test.npy    # [N_test × 2]   `

You can now evaluate any model using these files.

Testing Pretrained Models
-------------------------

### 1\. Pure GNN Baseline

*   **Location:** GNN model/ & checkpoints\_better/
    
*   **Checkpoint:** checkpoints\_better/better\_gnn.pth (included)
    

Test:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython "GNN model/test_better_gnn.py" \    --data_dir ./Preprocessed \    --checkpoint checkpoints_better/better_gnn.pth \    --batch_size 32   `

Prints precision, recall, F1 for arrhythmia classification.

### 2\. Pure Transformer Baseline

*   **Location:** Transformer\_model/ & Preprocessed/
    
*   **Checkpoints:** transformer\_fold{i}\_best.pth for i = 1..8 (included)
    

Test:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython "Transformer_model/test_kfold.py" \    --data_dir ./Preprocessed \    --checkpoint_dir ./Preprocessed \    --batch_size 32   `

Prints mean ± std of test-set F1 for arrhythmia (over 8 folds).

### 3\. Hybrid Transformer–GNN Model

*   **Location:** Hybrid model/
    
*   **Checkpoint:** Hybrid model/hyb\_try\_model\_ld.pth (included)
    

Arrhythmia Classification:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython test_hybrid.py \    --data_dir ./Preprocessed \    --checkpoint "Hybrid model/hyb_try_model_ld.pth" \    --batch_size 32   `

Stroke-Risk Regression:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   bashCopyEditpython "Hybrid model/run_finetune_ld.py" \    --data_dir ./Preprocessed \    --checkpoint "Hybrid model/hyb_try_model_ld.pth" \    --batch_size 32 \    --task regression   `

Reports Mean Squared Error (MSE) and Mean Absolute Error (MAE) for continuous regression.

Directory Structure
-------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   textCopyEdithybrid-ecg-thesis-code/  │  ├── preprocess.py  ├── Preprocessed/  │   ├── X_train.npy, X_val.npy, X_test.npy  │   ├── y_train.npy, y_val.npy, y_test.npy  │   ├── meta_train.npy, meta_val.npy, meta_test.npy  │   ├── classes.npy  │   ├── transformer_fold1_best.pth  │   ├── ... up to transformer_fold8_best.pth  ├── checkpoints_better/  │   └── better_gnn.pth  ├── GNN model/  │   ├── conv_rgnn_dataset.py  │   ├── better_spatial_gnn.py  │   ├── train_better_gnn.py  │   └── test_better_gnn.py  ├── Transformer_model/  │   ├── transformer_model_new.py  │   ├── transformer_train_kfolds.py  │   └── test_kfold.py  ├── Hybrid model/  │   ├── dataset_leaddrop.py  │   ├── models.py  │   ├── train_leaddrop.py  │   ├── run_finetune_ld.py  │   ├── train.py  │   └── hyb_try_model_ld.pth  ├── test_hybrid.py  └── README.md   `

Required Python Imports
-----------------------

Here is a reference for all external and standard library imports you may find across scripts in this repository:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pythonCopyEdit# Core  import os  import argparse  import glob  import math  import random  import time  import json  import logging  # Data science & processing  import numpy as np  import pandas as pd  import wfdb  # Machine Learning  from sklearn.model_selection import train_test_split, KFold  from sklearn.preprocessing import StandardScaler  from sklearn.metrics import (precision_score, recall_score, f1_score,                               mean_squared_error, mean_absolute_error)  # Torch & Deep Learning  import torch  import torch.nn as nn  import torch.nn.functional as F  from torch.utils.data import DataLoader, Dataset  # PyTorch Geometric (GNNs)  from torch_geometric.data import Data, InMemoryDataset  from torch_geometric.loader import DataLoader as GeoDataLoader  from torch_geometric.nn import MixHop, GCNConv, global_mean_pool  # Progress Bar / Utils  from tqdm import tqdm  import matplotlib.pyplot as plt   `

Ensure your environment includes all of these libraries. Some (such as torch-scatter, torch-sparse, torch-cluster, torch-geometric) require [special install instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

Citation
--------

If you use or extend this code, please cite:

> **Kumar Sambhavit**Hybrid Transformer–GNN Architecture for 12-Lead ECG Arrhythmia & Stroke-Risk Prediction.BSc Thesis, Department of Advanced Computing Sciences, Maastricht University, 2025.

Thank you for using this repository!If you encounter any issues or have questions, feel free to open an issue:[https://github.com/ksambhavit/hybrid-ecg-thesis-code](https://github.com/ksambhavit/hybrid-ecg-thesis-code)