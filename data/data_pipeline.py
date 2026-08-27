import pandas as pd
from pathlib import Path 
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter


def load_opportunity(files_path, add_group_id = False) -> pd.DataFrame:
    '''
    Read multiple .dat files from the Opportunity dataset and vertically concatenate
    them into a single DataFrame.

    If add_group_id=True, append a last column called 'group_id'
    with the session/file name, e.g. S1-ADL5.
    '''
    dataframes = []
    row_counts = 0

    for file in files_path:
        df = pd.read_csv(file, sep=r'\s+', header=None, engine='python', na_values='NaN')

        if add_group_id:
            session_name = Path(file).stem   # e.g. S1-ADL5
            df["group_id"] = session_name    # added as LAST column

        dataframes.append(df)
        row_counts += len(df)

    complete_dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

    assert len(complete_dataframe) == row_counts, (
        f"Row mismatch: complete_dataframe={len(complete_dataframe)} vs row_counts={row_counts}")
    return complete_dataframe
    
def remove_zero_label_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove rows where the label is 0.
    Works for:
      - train: label in last column
      - train/val/test: label in second-to-last column if last column is 'group_id'
    '''
    label_col_idx = -2 if dataframe.columns[-1] == "group_id" else -1
    zero_label = dataframe.iloc[:, label_col_idx] == 0
    return dataframe[~(zero_label)]

def remove_all_nan_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove any row with any NaN value. 
    (No interpolation done in the dataset later)
    '''
    mask_nans = dataframe.isna().any(axis = 1)
    return dataframe.loc[~mask_nans].copy()
    
def incomplete_labeled_rows(dataframe: pd.DataFrame) -> int:
    '''
    Count rows with at least one NaN in sensor columns only.
    Excludes:
      - label column in train
      - label + group_id in val/test/train
    '''
    sensor_data = dataframe.iloc[:, :-2] if dataframe.columns[-1] == "group_id" else dataframe.iloc[:, :-1]
    incomplete_rows = sensor_data.isna().any(axis=1)
    return int(incomplete_rows.sum())

def divide_features_labels(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # last mod for the windows per session to avoid creating art. windows change this in SP as well. what a mess
    if dataframe.columns[-1] == "group_id":
        sensor_ft = dataframe.iloc[:, :-2]
        group_id = dataframe[["group_id"]] #pd not series
        features = pd.concat([sensor_ft, group_id], axis = 1)
        labels = dataframe.iloc[:, -2]
    else:
        features = dataframe.iloc[:, :-1]
        labels = dataframe.iloc[:, -1]
    return features, labels

def acc_data_scaling (dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Relative magnitude differ across sensor types.
    For ACC data which is in mili g, this is divided by 1000 to obtain g, bringing values to a smaller range.
    '''
    dataframe_ = dataframe.copy()
    acc_cols = [axi + offset for offset in range(0, 45, 9) for axi in range(3)]
    dataframe_.iloc[:, acc_cols] /= 1000
    return dataframe_ 

def gyro_data_scaling (dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    GYRO data is milli rad/s, bringing values to a smaller range: divide by 1000 to obtain rad/s.
    '''
    dataframe_ = dataframe.copy()
    gyro_cols = [axi + offset for offset in range(3, 45, 9) for axi in range(3)]
    dataframe_.iloc[:, gyro_cols] /= 1000
    return dataframe_

def mag_data_norm (dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    For MAG the magnitude equation is used to remove the strenght of the signal and mantain the direction, which 
    tells us about the body orientation. 
    '''
    dataframe_ = dataframe.copy()
    for offset in range (6, 45, 9):
        mag_x = dataframe_.iloc[:, offset]
        mag_y = dataframe_.iloc[:, offset + 1]
        mag_z = dataframe_.iloc[:, offset + 2]
        magnitude = np.sqrt((mag_x ** 2) + (mag_y ** 2) + (mag_z ** 2))
        for axi in range(3):
            new_axi = dataframe_.iloc[:, axi + offset] / magnitude
            dataframe_.iloc[:, axi + offset] = new_axi
    return dataframe_

def sliding_window(X, y, window_size: int, stride: int, return_numpy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Row-wise sliding windows and assign a label policy: majority vote.
    Returns: features_windows: [#windows, window_size, #features]
             labels_windows: [#windows]
    '''

    
    X_is_df = isinstance(X, pd.DataFrame)
    y_is_series = isinstance(y, (pd.Series, pd.DataFrame))
    X_values = X.values if X_is_df else np.asarray(X)
    y_values = y.values.ravel() if y_is_series else np.asarray(y).ravel()

    num_rows, num_feats = X_values.shape
    X_windows = []
    y_windows = []
    window_count = 0
    # Slide Window
    for start in range(0, num_rows - window_size + 1, stride):
        end = start + window_size
        Xw = X_values[start:end, :]
        yw = y_values[start:end]

        win_label = Counter(yw).most_common(1)[0][0] # counts times each label appears. Take the tuple. Take the label (first position)
        X_windows.append(Xw)
        y_windows.append(int(win_label))
        window_count += 1

    # Stack Outputs
    if len(X_windows) == 0:
        if return_numpy:
            return np.empty((0, window_size, num_feats), dtype=np.float32), np.empty((0,), dtype=np.int64)
        else:
            return [], []

    if return_numpy:
        X_windows = np.stack(X_windows).astype(np.float32)  # [W,L,F]
        y_windows = np.asarray(y_windows, dtype=np.int64)   # [W]
    
    return X_windows, y_windows

#new function for group_id
def sliding_window_wrapper_group(X, y, window_size: int, stride: int)-> Tuple[np.ndarray, np.ndarray]:
    
    '''Wrapper to create sliding windows per group_id to avoid creating artificial windows among sessions.'''
    
    X_windows_all = []
    y_windows_all = []
    
    for _, X_group in X.groupby("group_id"):
        idx_group = X_group.index
        y_group = y.loc[idx_group] #filter labels
        X_group = X_group.drop(columns=["group_id"]) #data with no group

        X_windows, y_windows = sliding_window(X_group, y_group, window_size=window_size, stride=stride)
        if len(X_windows) > 0:
            X_windows_all.append(X_windows)
            y_windows_all.append(y_windows)

    if len(X_windows_all) == 0: #same check as in sliding_window
        num_feats = X.drop(columns=["group_id"]).shape[1]
        return np.empty((0, window_size, num_feats), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_windows_all = np.concatenate(X_windows_all, axis = 0)
    y_windows_all = np.concatenate(y_windows_all, axis = 0)            
    return X_windows_all, y_windows_all 


'''def sliding_window(X, y, window_size: int, stride: int, min_purity: float = 1.0, return_numpy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
 
    Row-wise sliding windows with majority-vote labels.
    Only keeps windows where the majority label covers >= min_purity of the window.
    min_purity=1.0  -> strict, all samples must share the same label
    min_purity=0.95 -> allow up to 5% contamination (recommended starting point)
   
    X_is_df = isinstance(X, pd.DataFrame)
    y_is_series = isinstance(y, (pd.Series, pd.DataFrame))
    X_values = X.values if X_is_df else np.asarray(X)
    y_values = y.values.ravel() if y_is_series else np.asarray(y).ravel()
    num_rows, num_feats = X_values.shape
    X_windows = []
    y_windows = []
    dropped = 0

    for start in range(0, num_rows - window_size + 1, stride):
        end = start + window_size
        Xw = X_values[start:end, :]
        yw = y_values[start:end]

        # Majority label + purity check
        win_label, count = Counter(yw).most_common(1)[0]
        purity = count / window_size

        if purity < min_purity:
            dropped += 1
            continue  # skip contaminated window

        X_windows.append(Xw)
        y_windows.append(int(win_label))

    if len(X_windows) == 0:
        if return_numpy:
            return np.empty((0, window_size, num_feats), dtype=np.float32), np.empty((0,), dtype=np.int64)
        else:
            return [], []
    if return_numpy:
        X_windows = np.stack(X_windows).astype(np.float32)
        y_windows = np.asarray(y_windows, dtype=np.int64)

    return X_windows, y_windows


def sliding_window_wrapper_group(X, y, window_size: int, stride: int, min_purity: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    
    Wrapper to create sliding windows per group_id, skipping windows whose
    label purity is below `min_purity`.
    
    X_windows_all = []
    y_windows_all = []

    for _, X_group in X.groupby("group_id"):
        idx_group = X_group.index
        y_group = y.loc[idx_group]
        X_group = X_group.drop(columns=["group_id"])
        X_windows, y_windows = sliding_window(
            X_group, y_group,
            window_size=window_size,
            stride=stride,
            min_purity=min_purity,
        )
        if len(X_windows) > 0:
            X_windows_all.append(X_windows)
            y_windows_all.append(y_windows)

    if len(X_windows_all) == 0:
        num_feats = X.drop(columns=["group_id"]).shape[1]
        return np.empty((0, window_size, num_feats), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X_windows_all = np.concatenate(X_windows_all, axis=0)
    y_windows_all = np.concatenate(y_windows_all, axis=0)
    return X_windows_all, y_windows_all

def make_windows_with_report(X, y, split_name, window_size=150, stride=30, min_purity=0.95):
    # Before filtering: all majority-vote windows
    X_all, y_all = sliding_window_wrapper_group(
        X, y,
        window_size=window_size,
        stride=stride,
        min_purity=0.0
    )

    # After filtering: only pure windows
    X_filt, y_filt = sliding_window_wrapper_group(
        X, y,
        window_size=window_size,
        stride=stride,
        min_purity=min_purity
    )

    print(f"\n===== {split_name} =====")
    print(f"min_purity = {min_purity}")
    print("Before:", Counter(y_all))
    print("After: ", Counter(y_filt))
    print("Dropped:", len(y_all) - len(y_filt), f"({(len(y_all)-len(y_filt))/len(y_all):.2%})")

    return X_filt, y_filt '''
    
def mag_data_rotation(X_windows) -> np.array:
    '''
    Demeaning
    '''
    X_w = X_windows.copy()
    mag_cols = [axis + offset for offset in range(6, 45, 9) for axis in range(3)]
    for channel in mag_cols:
        mean_val = X_w[:, :, channel].mean(axis = 1, keepdims = True)  # [W, 1] for windows W compute mean in that column. 
        X_w[:, :, channel] -= mean_val  # remove orientation, keep rotation dynamics
    return X_w
        
    








