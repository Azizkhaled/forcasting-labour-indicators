from itertools import islice
import re
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from tqdm.notebook import tqdm
from lightning.pytorch.callbacks import EarlyStopping


import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')  # or 'medium'
from gluonts.evaluation import make_evaluation_predictions
import sys
sys.path.append('./lag-llama')
from lag_llama.gluon.estimator import LagLlamaEstimator


from scripts.data_processing import with_external_load_dataset, No_external_load_dataset

def get_lag_llama_predictions(dataset, prediction_length, model_weights_path = "models\lag-llama.ckpt",context_length=32, num_samples=20, device="cuda", batch_size=64, nonnegative_pred_samples=True):
    ckpt = torch.load(model_weights_path, map_location=device)
    # ckpt = torch.load(r"lightning_logs\version_7\checkpoints\epoch=19-step=1000.ckpt", map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path="models\lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length,

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],

        nonnegative_pred_samples=nonnegative_pred_samples,

        # linear positional encoding scaling
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        },

        batch_size=batch_size,
        num_parallel_samples=num_samples,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

    return forecasts, tss

def lama_tune(starting_weights_path="models\lag-llama.ckpt", device = "cuda", prediction_length = 11, context_length_factor = 6, num_samples = 100, max_epochs = 50, batch_size = 64, lr = 5e-4):
    ckpt = torch.load(starting_weights_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    early_stopping_callback = EarlyStopping(
    monitor='val_loss',  
    patience=15,        
    verbose=True,
    mode='min'           
    )
    estimator = LagLlamaEstimator(
            ckpt_path=starting_weights_path,
            prediction_length=prediction_length,
            context_length=prediction_length * context_length_factor,

            # distr_output="neg_bin",
            # scaling="mean",
            track_loss_per_series= True,
            nonnegative_pred_samples=True,
            aug_prob=0,
            lr=lr,

            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            time_feat=estimator_args["time_feat"],

            # rope_scaling={
            #     "type": "linear",
            #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
            # },

            

            batch_size=batch_size,
            num_parallel_samples=num_samples,
            trainer_kwargs = {"max_epochs": max_epochs,
                              'callbacks': [early_stopping_callback]}, # <- lightning trainer arguments
        )
    return estimator

class llama_models:
    def __init__(self, path_to_csv, lr = 5e-4, prediction_length = 11, batch_size =64, num_samples = 100, context_length_factor = 6, device = "cuda", with_external = True, selected_features = None):

        self.prediction_length = prediction_length
        self.num_samples = num_samples  # number of samples sampled from the probability distribution for each timestep
        self.context_length_factor = context_length_factor
        self.path_to_csv = path_to_csv
        self.context_length = self.prediction_length * self.context_length_factor
        self.device = device
        self.batch_size=batch_size
        self.train_earnings_dataset, self.val_earnings_dataset, self.test_earnings_dataset = with_external_load_dataset(self.path_to_csv, selected_features) if with_external else No_external_load_dataset(self.path_to_csv)
        self.lr = lr

    
    def baseline_llama_on_csv(self,  model_weights_path = "models\lag-llama.ckpt"):
        forecasts, tss = get_lag_llama_predictions(
        model_weights_path = model_weights_path,
        dataset= self.test_earnings_dataset,
        prediction_length= self.prediction_length,
        num_samples= self.num_samples,
        context_length= self.context_length,
        device= self.device,
        batch_size=self.batch_size
        )
        return forecasts, tss
    
    def fine_tuned_model(self, initial_weights_path = "models\lag-llama.ckpt", max_epochs = 50):
        estimator = lama_tune(
            starting_weights_path= initial_weights_path,
            device = self.device,
            prediction_length = self.prediction_length,
            context_length_factor = self.context_length_factor,
            num_samples = self.num_samples,
            max_epochs= max_epochs,
            batch_size=self.batch_size,
            lr=self.lr
            )
        earnings_predictor = estimator.train(training_data=self.train_earnings_dataset, validation_data= self.val_earnings_dataset, cache_data=True, shuffle_buffer_length=1000)
        forecast_it, ts_it = make_evaluation_predictions(
        dataset=self.test_earnings_dataset,
        predictor=earnings_predictor,
        num_samples=self.num_samples
            )

        forecasts = list(tqdm(forecast_it, total=len(self.test_earnings_dataset), desc="Forecasting batches"))
        tss = list(tqdm(ts_it, total=len(self.test_earnings_dataset), desc="Ground truth"))

        return forecasts,tss
    
    