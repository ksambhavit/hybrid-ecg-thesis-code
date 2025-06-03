# src/preprocess.py
import os
import ast
import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer

def aggregate_diagnostic(y_dic, agg_df):
    """Aggregate diagnostic classes from scp_codes for each record."""
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def load_raw_data_batch(df, sampling_rate, path, batch_size=200):
    """
    Generator to load ECG data in batches (batch_size records at a time)
    to reduce memory usage.
    """
    num_samples = len(df)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]
        if sampling_rate == 100:
            data = [wfdb.rdsamp(os.path.join(path, f)) for f in batch_df.filename_lr]
        else:
            data = [wfdb.rdsamp(os.path.join(path, f)) for f in batch_df.filename_hr]
        yield np.array([signal for signal, meta in data])

def compute_global_stats(df, sampling_rate, data_path, sample_size=2000):
    """Compute global mean/std from a sample to normalize data consistently."""
    subset_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    all_signals = []
    for signals in load_raw_data_batch(subset_df, sampling_rate, data_path, batch_size=200):
        all_signals.append(signals)
        if sum(x.shape[0] for x in all_signals) >= sample_size:
            break
    all_signals = np.concatenate(all_signals, axis=0)
    global_mean = all_signals.mean()
    global_std = all_signals.std() + 1e-8
    return global_mean, global_std

def preprocess_ptbxl(data_path, sampling_rate=500, batch_size=200, output_dir="../processed_data/", compute_global_norm=True, sample_size_for_stats=2000):
    """
    Preprocess PTB-XL data in batches.
    Saves X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test, and classes as .npy files.
    """
    # 1. Load metadata
    Y = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # 2. Load diagnostic info and aggregate labels
    agg_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))
    
    # 3. MultiLabelBinarizer for 5 classes (adjust if needed)
    mlb = MultiLabelBinarizer(classes=['NORM', 'MI', 'STTC', 'CD', 'HYP'])
    labels = mlb.fit_transform(Y['diagnostic_superclass'])
    
    # 4. Process metadata (age, sex)
    metadata = Y[['age', 'sex']].copy()
    metadata['age'] = (metadata['age'] - metadata['age'].mean()) / (metadata['age'].std() + 1e-8)
    metadata['sex'] = metadata['sex'].map({0: 0, 1: 1})
    metadata = metadata.to_numpy()
    
    # 5. Split indices based on folds (train: folds 1-8, val: fold 9, test: fold 10)
    train_mask = Y['strat_fold'].isin(range(1, 9))
    val_mask = Y['strat_fold'] == 9
    test_mask = Y['strat_fold'] == 10
    
    # 6. Compute global normalization stats if requested
    global_mean, global_std = 0.0, 1.0
    if compute_global_norm:
        print(f"Computing global mean/std from ~{sample_size_for_stats} records. Please wait...")
        global_mean, global_std = compute_global_stats(Y, sampling_rate, data_path, sample_size=sample_size_for_stats)
        print(f"Global mean: {global_mean:.4f}, Global std: {global_std:.4f}")
    
    # 7. Process ECG data in batches and split into train/val/test
    train_data, val_data, test_data = [], [], []
    total_records = len(Y)
    for batch_idx, batch_ecgs in enumerate(load_raw_data_batch(Y, sampling_rate, data_path, batch_size)):
        if compute_global_norm:
            batch_ecgs = (batch_ecgs - global_mean) / global_std
        else:
            batch_ecgs = (batch_ecgs - batch_ecgs.mean()) / (batch_ecgs.std() + 1e-8)
        
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_records)
        batch_train = batch_ecgs[train_mask[start_idx:end_idx].values]
        batch_val = batch_ecgs[val_mask[start_idx:end_idx].values]
        batch_test = batch_ecgs[test_mask[start_idx:end_idx].values]
        
        if batch_train.size > 0:
            train_data.append(batch_train)
        if batch_val.size > 0:
            val_data.append(batch_val)
        if batch_test.size > 0:
            test_data.append(batch_test)
    
    X_train = np.concatenate(train_data, axis=0) if train_data else np.array([])
    X_val = np.concatenate(val_data, axis=0) if val_data else np.array([])
    X_test = np.concatenate(test_data, axis=0) if test_data else np.array([])
    
    # 8. Split labels and metadata
    y_train, meta_train = labels[train_mask], metadata[train_mask]
    y_val, meta_val = labels[val_mask], metadata[val_mask]
    y_test, meta_test = labels[test_mask], metadata[test_mask]
    
    # 9. Save outputs to disk
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(output_dir, "meta_train.npy"), meta_train)
    np.save(os.path.join(output_dir, "meta_val.npy"), meta_val)
    np.save(os.path.join(output_dir, "meta_test.npy"), meta_test)
    np.save(os.path.join(output_dir, "classes.npy"), mlb.classes_)
    
    print(f"Final shapes:\n"
          f"X_train={X_train.shape}, y_train={y_train.shape}, meta_train={meta_train.shape}\n"
          f"X_val={X_val.shape},   y_val={y_val.shape},   meta_val={meta_val.shape}\n"
          f"X_test={X_test.shape}, y_test={y_test.shape}, meta_test={meta_test.shape}\n")
    return X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test, mlb.classes_

if __name__ == "__main__":
    data_path = "./Data/"  # Update to your PTB-XL data folder location
    preprocess_ptbxl(
        data_path=data_path,
        sampling_rate=500,
        batch_size=200,
        output_dir="./processed_data/",
        compute_global_norm=True,
        sample_size_for_stats=2000
    )
    print("Preprocessing complete!")
