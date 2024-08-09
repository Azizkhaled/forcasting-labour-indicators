from itertools import islice
import re
import os
import pandas as pd

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is: ", device)


def convertcolumns_tofloat(df):
    for col in df.columns:
        # Check if column is not of string type
        if df[col].dtype != 'object' and pd.api.types.is_string_dtype(df[col]) == False:
            df[col] = df[col].astype('float32')


def mean_abs_scaling(context, min_scale=1e-5):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)


def custom_collate_fn(batch):
    past_targets = []
    future_targets = []
    dynamic_features = []
    # print("BATCH INDICATOR ##############################################################################################################################################")
    for item in batch:
        # Assume each 'item' in the batch has 'time_series', and 'dynamic_features'
        time_series = torch.tensor(item['target'], dtype=torch.float).to(device)  # This should be your full series data
        features = torch.tensor(item['feat_dynamic_real'], dtype=torch.float).to(device)  # Your dynamic features for each time point

        # Define how far back you look and how far ahead you predict
        prediction_length = 11  # For example, predict the next 5 data points
        context_length = prediction_length * 6
        # print(time_series[-(context_length+prediction_length):-prediction_length])
        # print(time_series[-prediction_length:])
        # print(sdgsdg)
        # print("full:",time_series.shape)
        # print("future:",time_series[-prediction_length:].shape)
        # print("past:",time_series[context_length:-prediction_length].shape)
        past_targets.append(time_series[-(context_length+prediction_length):-prediction_length])
        future_targets.append(time_series[-prediction_length:])
        dynamic_features.append(features)

    # Convert lists to tensors
    past_targets_batch = torch.stack(past_targets).to(device)
    future_targets_batch = torch.stack(future_targets).to(device)
    dynamic_features_batch = torch.stack(dynamic_features).to(device)
    # Use 'batchify' to stack your arrays/tensors into batches
    # past_targets_batch = batchify(past_targets)
    # future_targets_batch = batchify(future_targets)
    
    # If you're using dynamic features
    if 'feat_dynamic_real' in batch[0]:
        # dynamic_features_batch = batchify(dynamic_features)
        return {'past_target': past_targets_batch, 'future_target': future_targets_batch, 'feat_dynamic_real': dynamic_features_batch}

    return {'past_target': past_targets_batch, 'future_target': future_targets_batch}


# used for business outlook survey
def convert_to_date(qtr):
    year = int(qtr[:4])
    if qtr[-2:] == 'Q4':
        year += 1
    month = {
        'Q1': '04',
        'Q2': '07',
        'Q3': '10',
        'Q4': '01'
    }[qtr[-2:]]
    return f"{year}-{month}-01"