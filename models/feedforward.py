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

import sys
sys.path.append('./lag-llama')

from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import StudentTOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator
import lightning.pytorch as pl
from typing import List, Callable
from gluonts.dataset.loader import TrainDataLoader,ValidationDataLoader

from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    InstanceSplitter,
    TestSplitSampler,
)
import math


from scripts.data_processing import with_external_load_dataset, No_external_load_dataset
from scripts.utils import (mean_abs_scaling, custom_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is: ", device)

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dimensions: List[int],
        distr_output=StudentTOutput(),
        batch_norm: bool = False,
        scaling: Callable = mean_abs_scaling,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert len(hidden_dimensions) > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimensions = hidden_dimensions
        self.distr_output = distr_output
        self.batch_norm = batch_norm
        self.scaling = scaling

        dimensions = [context_length] , hidden_dimensions[:-1]

        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [self.__make_lin(in_size, out_size), nn.ReLU()]
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_size))
        modules.append(
            self.__make_lin(dimensions[-1], prediction_length * hidden_dimensions[-1])
        )

        self.nn = nn.Sequential(*modules)
        self.args_proj = self.distr_output.get_args_proj(hidden_dimensions[-1])

    @staticmethod
    def __make_lin(dim_in, dim_out):
        lin = nn.Linear(dim_in, dim_out)
        torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
        torch.nn.init.zeros_(lin.bias)
        return lin

    def forward(self, context):
        scale = self.scaling(context)
        scaled_context = context / scale
        nn_out = self.nn(scaled_context)
        nn_out_reshaped = nn_out.reshape(
            -1, self.prediction_length, self.hidden_dimensions[-1]
        )
        distr_args = self.args_proj(nn_out_reshaped)
        return distr_args, torch.zeros_like(scale), scale

    def get_predictor(self, input_transform, batch_size=32):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=["past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            forecast_generator=DistributionForecastGenerator(self.distr_output),)
    
class LightningFeedForwardNetwork(FeedForwardNetwork, pl.LightningModule):
    def __init__(self,lr , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.to(device)

    def training_step(self, batch):
        context = batch["past_target"].to(device)
        target = batch["future_target"].to(device)
        context[torch.isnan(context)] = 0
        target[torch.isnan(target)] = 0
        assert context.shape[-1] == self.context_length
        assert target.shape[-1] == self.prediction_length
        distr_args, loc, scale = self(context)
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)
        return loss.mean()
    
    def validation_step(self, batch):
            context = batch["past_target"].to(device)
            target = batch["future_target"].to(device)
            context[torch.isnan(context)] = 0
            target[torch.isnan(target)] = 0

            assert context.shape[-1] == self.context_length
            assert target.shape[-1] == self.prediction_length

            distr_args, loc, scale = self(context)
            distr = self.distr_output.distribution(distr_args, loc, scale)
            loss = -distr.log_prob(target)

            # Optionally log validation loss using log_dict
            self.log('val_loss', loss.mean(), on_step=False, on_epoch=True, prog_bar=True)
            
            return {'val_loss': loss.mean()}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def run_feed_forward_model(path_to_csv, prediction_length = 11, context_length_factor = 6, hidden_dimensions = [96, 48], max_epochs = 50, batch_size=64, lr = 5e-4, with_external = True, selected_features= None):

    train_earnings_dataset, val_earnings_dataset, test_earnings_dataset = with_external_load_dataset(path_to_csv, selected_features=selected_features) if with_external else No_external_load_dataset(path_to_csv)

    length_of_dataset = sum(1 for _ in train_earnings_dataset)

    num_batches_per_epoch = math.ceil(length_of_dataset / batch_size)
    print(length_of_dataset)
    print(f"Number of training batches per epoch: {num_batches_per_epoch}")     

    train_data_loader = TrainDataLoader(train_earnings_dataset, batch_size=batch_size,num_batches_per_epoch=num_batches_per_epoch, stack_fn=custom_collate_fn)

    val_data_loader = ValidationDataLoader(val_earnings_dataset, batch_size=batch_size, stack_fn=custom_collate_fn)

    context_length = prediction_length * context_length_factor

    net = LightningFeedForwardNetwork(lr,
            prediction_length=prediction_length,
            context_length=context_length,
            hidden_dimensions=hidden_dimensions,
            distr_output=StudentTOutput())
    
    early_stopping_callback = EarlyStopping(
    monitor='val_loss',  
    patience=15,        
    verbose=False,
    mode='min'           
    )
    
    trainer = pl.Trainer(max_epochs=max_epochs,callbacks = early_stopping_callback)

    trainer.fit(model = net, train_dataloaders = train_data_loader,val_dataloaders=val_data_loader)

    prediction_splitter = InstanceSplitter(
    target_field=FieldName.TARGET,
    is_pad_field=FieldName.IS_PAD,
    start_field=FieldName.START,
    forecast_start_field=FieldName.FORECAST_START,
    instance_sampler=TestSplitSampler(),
    past_length=context_length,
    future_length=prediction_length,
    time_series_fields=[FieldName.OBSERVED_VALUES],)

    mask_unobserved = AddObservedValuesIndicator(
    target_field=FieldName.TARGET,
    output_field=FieldName.OBSERVED_VALUES,
    )

    predictor_pytorch = net.get_predictor(mask_unobserved + prediction_splitter)

    forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_earnings_dataset,
    predictor=predictor_pytorch
    )

    forecasts = list(tqdm(forecast_it, total=len(test_earnings_dataset), desc="Forecasting batches",miniters=10))
    tss = list(tqdm(ts_it, total=len(test_earnings_dataset), desc="Ground truth",miniters=10))

    return forecasts,tss