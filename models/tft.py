from itertools import islice
import re
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from tqdm.notebook import tqdm
from lightning.pytorch.callbacks import EarlyStopping


import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')  # or 'medium'
from gluonts.evaluation import make_evaluation_predictions
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator

import math


from scripts.data_processing import with_external_load_dataset, No_external_load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is: ", device)


def run_tft(path_to_csv, prediction_length = 11, context_length_factor = 6, max_epochs = 50, batch_size=64, lr = 5e-4, with_external= True, selected_features= None):

    train_earnings_dataset, val_earnings_dataset, test_earnings_dataset = with_external_load_dataset(path_to_csv, selected_features=selected_features) if with_external else No_external_load_dataset(path_to_csv)

    length_of_dataset = sum(1 for _ in train_earnings_dataset)

    num_batches_per_epoch = math.ceil(length_of_dataset / batch_size)
    print(length_of_dataset)
    print(f"Number of training batches per epoch: {num_batches_per_epoch}")       

    context_length = prediction_length * context_length_factor

    estimator = TemporalFusionTransformerEstimator(
        freq='1M',
        prediction_length=prediction_length,
        context_length=context_length,
        num_heads=4,
        hidden_dim=32,
        variable_dim=32,
        lr = lr,
        batch_size=batch_size
        )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  
        patience=10,        
        verbose=False,
        mode='min',
        # log_rank_zero_only=True          
        )
#     checkpoint_callback = ModelCheckpoint(
#     monitor='val_loss',
#     save_top_k=1,
#     verbose=False,  # Turn off messages about checkpoints
# )
        
    predictor = estimator.train(train_earnings_dataset,val_earnings_dataset,max_epochs=max_epochs,callbacks = early_stopping_callback)

    forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_earnings_dataset,
    predictor=predictor
    )

    forecasts = list(tqdm(forecast_it, total=len(test_earnings_dataset), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(test_earnings_dataset), desc="Ground truth"))

    return forecasts,tss