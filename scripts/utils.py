'''
----------------------
Evan Zelt & Brett Westerberg
CSC 373: Assignment 5
Dr. Khuri
Wake Forest University
3/26/2025
----------------------

Module for utility functions
Acknowledgements: 
    - Dr. Khuri (Wake Forest University)

'''
import os
import gzip
import pandas as pd
import numpy as np

def get_dataset():
    file_path = "/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz"
    input_file = gzip.open(file_path)
    
    dataset = []
    
    for l in input_file:
        d = eval(l)
        dataset.append(d)

    input_file.close()
    
    return dataset


def get_subsample():
    file_path = "../data/subsample.csv"
    return pd.read_csv(file_path)


def get_dataset_split(dataset):
    train_data = dataset[:int(len(dataset)*0.8)]
    dev_data = dataset[int(len(dataset)*0.8):]

    return train_data, dev_data


def get_dates(dataset):
    dates = []
    for i in range(len(dataset)):
        dates.append(int(dataset[i]['date'][:4]))

    return dates


def create_df(dataset):
    return pd.DataFrame(dataset)


def subsample_df(df, output_file="../data/subsample.csv", n=10000):
    sample = df.sample(n)
    sample.to_csv(output_file)

    return sample
    

def report_df(df, output_dir="../output/data_reports"):
    output_path = os.path.join(output_dir, "dataset_summary.txt")
    handle = open(output_path, "w+")

    handle.write("Basic dataset information:\n")
    num_entries = len(df)
    handle.write(f"Number of entries: {num_entries}\n\n")
    
    handle.write("Numerical Descriptive Statistics:\n")
    desc_stats = df.describe()
    handle.write(f"{desc_stats.to_string()}\n\n")

    handle.write("Missing (NA) Counts:\n")
    missing_counts_cols = df.isna().sum()
    handle.write(f"{missing_counts_cols.to_string()}\n\n")

    handle.write("Columns with many missing values (over 10%):\n") 
    many_missing_cols = missing_counts_cols[missing_counts_cols > num_entries * 0.1]
    handle.write(f"{many_missing_cols.to_string()}\n\n")
    
    handle.write("Rows with many missing values:\n")
    missing_counts_rows = df.isna().sum(axis=1)
    num_cols = len(df.columns)
    many_missing_rows = missing_counts_rows[missing_counts_rows > 0.8 * num_cols]
    handle.write(f"{many_missing_rows.to_string()}\n\n")


    handle.close()


def clean_data(df):
    sparse_cols = df.columns[df.isna().sum() > 0.1 * len(df)]
    df = df.drop(columns=sparse_cols)

    df[['year', 'month', 'day']] = df['date'].str.split('-', expand=True)
    df[['year', 'month', 'day']] = df[['year', 'month', 'day']].apply(pd.to_numeric)

    df = df.drop(columns=['username', 'date'])

    df = df.dropna()
    df = df.drop_duplicates()

    return df


def create_target(df):
    y = df['hours']
    X = df.drop(columns=['hours'])
    
    return X, y


def sort_cols(df):
    num_col_selector = (df.dtypes == "int64") | (df.dtypes == "float64")
    text_col_selector = df.dtypes == "object"

    num_cols = df.columns[num_col_selector]
    text_cols = df.columns[text_col_selector]
    cat_cols = df.columns[~(num_col_selector | text_col_selector)]

    return num_cols, text_cols, cat_cols


def over_under_pred_scores(trues, preds, classification=False):
    total = len(trues)

    if classification:
        over_pred_score = sum((preds == 1) & (trues == 0)) / total
        under_pred_score = sum((preds == 0) & (trues == 1)) / total        

        return over_pred_score, under_pred_score

    over_pred_score = sum(preds > trues) / total
    under_pred_score = sum(preds < trues) / total

    return over_pred_score, under_pred_score


def get_log_y(y):
    return np.log2(y+1)


def write_estimation_scores(handle, scores):
    handle.write(f"\tR-Squared: {scores['r2']:.3f}\n")
    handle.write(f"\tMSE (dev): {scores['mse']:.3f}\n")
    handle.write(f"\tOverprediction Rate (dev): {scores['over']:.2f}\n")
    handle.write(f"\tUnderprediction Rate (dev): {scores['under']:.2f}\n\n")


def write_classification_scores(handle, scores):
    handle.write(f"\tAccurracy (dev): {scores['accuracy']:.3f}\n")
    handle.write(f"\tOverprediction Rate (dev): {scores['over']:.2f}\n")
    handle.write(f"\tUnderprediction Rate (dev): {scores['under']:.2f}\n\n")


def binarize_y(y):
    return (y > y.median()).astype(int)


def remove_outliers(X, y, a=.1):
    combined = X.copy()
    combined['hours'] = y

    threshold = combined['hours'].quantile(1 - a)
    filtered = combined[combined['hours'] < threshold]

    y = filtered['hours']
    X = filtered.drop(columns=['hours'])

    return X, y
