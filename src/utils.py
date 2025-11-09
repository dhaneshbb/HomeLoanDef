import gc
import os

import numpy as np
import pandas as pd
import psutil


def memory_usage():
    """Prints the current memory usage of the Python process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")


def dataframe_memory_usage():
    """Prints the memory usage of all loaded Pandas DataFrames."""
    datasets = {
        "application_train": at,
        "bureau": bu,
        "bureau_balance": bub,
        "POS_CASH_balance": pc,
        "credit_card_balance": ccb,
        "previous_application": pa,
        "installments_payments": ip,
    }
    for name, df in datasets.items():
        mem_usage = df.memory_usage(deep=True).sum() / 1024**2
        print(f"{name} Memory Usage: {mem_usage:.2f} MB")


def garbage_collection():
    """Performs garbage collection to free up memory."""
    gc.collect()
    memory_usage()


def single_value_columns(df):
    total_entries = df.shape[0]
    single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
    if not single_value_cols:
        print("No columns with a single unique value found.")
        return None
    single_value_summary = pd.DataFrame(
        {
            "Index": range(len(single_value_cols)),
            "Column Name": single_value_cols,
            "Data Type": [df[col].dtype for col in single_value_cols],
            "Missing Percentage": [
                (df[col].isna().sum() / total_entries) * 100
                for col in single_value_cols
            ],
        }
    )
    return single_value_summary


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df


def reduce_mem_usagewithout_causing_cat(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe before optimization: {start_mem:.2f} MB")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type in ["int64", "int32", "int16", "int8"]:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        elif col_type in ["float64", "float32", "float16"]:
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif col_type.name == "category":
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization: {end_mem:.2f} MB")
    print(f"Memory reduction: {100 * (start_mem - end_mem) / start_mem:.2f}%")
    return df


def optimize_data_types(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def cap_extreme_values(df, columns):
    for col in columns:
        percentile_99 = df[col].quantile(0.99)
        df[col] = np.minimum(df[col], percentile_99)
    return df


def cap_outliers(df, columns, cap_value):
    for col in columns:
        df[col] = df[col].apply(
            lambda x: min(x, cap_value)
        )  # Cap values at the given threshold
    return df


def cap_outliers(df, columns, percentile=0.99):
    for col in columns:
        cap_value = df[col].quantile(percentile)
        df[col] = np.minimum(df[col], cap_value)
    return df
