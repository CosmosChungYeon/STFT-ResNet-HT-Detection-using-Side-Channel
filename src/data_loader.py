import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

AES_VERSIONS = ['AES-T500', 'AES-T600', 'AES-T700', 'AES-T800', 'AES-T1600']
IS_RAND = 1  # Random: 1, Fixed: 2
RAW_DATASET_PATH = "./OpenDataset_CSV"
DATASET_PATH = "./OpenDataset_NPY"
DATASET_PATH_CW305 = "./CW305_traces"

def csv_to_npy(versions=AES_VERSIONS, n_samples=10000, points=2500):
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    for version in tqdm(versions, desc="CSV→NPY"):
        disabled = np.zeros((n_samples, points), dtype=np.int32)
        d_dir = f"{RAW_DATASET_PATH}/{version}_power_Temp25C/{version}+TrojanDisabled_{IS_RAND}/{version}+TrojanDisabled_{IS_RAND}"
        enabled = np.zeros((n_samples, points), dtype=np.int32)
        e_dir = f"{RAW_DATASET_PATH}/{version}_power_Temp25C/{version}+TrojanEnabled_{IS_RAND}/{version}+TrojanEnabled_{IS_RAND}"
        triggered = np.zeros((n_samples, points), dtype=np.int32)
        t_dir = f"{RAW_DATASET_PATH}/{version}_power_Temp25C/{version}+TrojanTriggered_{IS_RAND}/{version}+TrojanTriggered_{IS_RAND}"
        
        for i in range(n_samples):
            d_csv = pd.read_csv(f"{d_dir}/Sample_{i}.csv", header=None)
            disabled[i] = d_csv.values.flatten()
            e_csv = pd.read_csv(f"{e_dir}/Sample_{i}.csv", header=None)
            enabled[i] = e_csv.values.flatten()
            t_csv = pd.read_csv(f"{t_dir}/Sample_{i}.csv", header=None)
            triggered[i] = t_csv.values.flatten()

        # 저장
        np.save(f"{DATASET_PATH}/{version}-Disabled_{IS_RAND}.npy", disabled)
        np.save(f"{DATASET_PATH}/{version}-Enabled_{IS_RAND}.npy", enabled)
        np.save(f"{DATASET_PATH}/{version}-Triggered_{IS_RAND}.npy", triggered)

def load_trace(version="AES-T500"):
    normal_path    = f"{DATASET_PATH}/{version}-Disabled_{IS_RAND}.npy"
    triggered_path = f"{DATASET_PATH}/{version}-Triggered_{IS_RAND}.npy"

    if os.path.exists(normal_path) and os.path.exists(triggered_path):
        normal = np.load(normal_path)
        triggered = np.load(triggered_path)
        trace = np.concatenate((normal, triggered), axis=0)
        print(f"Data shape for {version}: {trace.shape}")
    else:
        print(f"'{version}' file is not found.")
        return None
    return trace

def load_trace_CW305(version="AES-T500"):
    # normal_path    = f"{DATASET_PATH_CW305}/AES_traces.npy"
    # triggered_path = f"{DATASET_PATH_CW305}/{version}_traces.npy"

    # if os.path.exists(normal_path) and os.path.exists(triggered_path):
    #     normal = np.load(normal_path)
    #     triggered = np.load(triggered_path)
    #     trace = np.concatenate((normal, triggered), axis=0)
    #     print(f"Data shape for {version}: {trace.shape}")
    # else:
    #     print(f"'{version}' file is not found.")
    #     return None
    path = f"{DATASET_PATH_CW305}/CW305_20000traces_400points_FK_traces_{version}.npy"

    if os.path.exists(path):
        trace = np.load(path)
        print(f"Data shape for {version}: {trace.shape}")
    else:
        print(f"'{version}' file is not found.")
        return None
    return trace

def load_supervised_set(version="AES-T500", load_path="./supervised_dataset"):
    X_train = np.load(f"{load_path}/{version}_train.npy")
    y_train = np.load(f"{load_path}/{version}_train_labels.npy")
    X_val   = np.load(f"{load_path}/{version}_val.npy")
    y_val   = np.load(f"{load_path}/{version}_val_labels.npy")
    X_test  = np.load(f"{load_path}/{version}_test.npy")
    y_test  = np.load(f"{load_path}/{version}_test_labels.npy")
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_supervised_set_KMU(version="AES-T500", load_path="./supervised_dataset_CW305/KMU"):
    X_train = np.load(f"{load_path}/KMU_{version}_train.npy")
    y_train = np.load(f"{load_path}/KMU_{version}_train_labels.npy")
    X_val   = np.load(f"{load_path}/KMU_{version}_val.npy")
    y_val   = np.load(f"{load_path}/KMU_{version}_val_labels.npy")
    X_test  = np.load(f"{load_path}/KMU_{version}_test.npy")
    y_test  = np.load(f"{load_path}/KMU_{version}_test_labels.npy")
    return X_train, y_train, X_val, y_val, X_test, y_test