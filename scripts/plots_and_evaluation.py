from itertools import islice
import re
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from gluonts.evaluation import  Evaluator

from models.feedforward import run_feed_forward_model
from models.tft import run_tft
from models.llama import llama_models
import logging
from logging.handlers import RotatingFileHandler

class plots_and_evaluation:
    
    def __init__(self, forecasts,tss, prediction_length = 11, title= '', plot_only = None):
        self.forecasts = forecasts
        self.tss = tss
        self.prediction_length = prediction_length
        self.title = title
        self.plot_only = plot_only
    
    def plot_forcasts_all(self, zoom_to_predicted = False, save_path = None):
        plt.figure(figsize=(10,7) if self.plot_only else (20, 27))  # Adjust the figure size as needed
        date_formater = mdates.DateFormatter('%Y,%m')
        plt.rcParams.update({'font.size': 9})
        # Iterate through all 22 series and plot the predicted samples
        for idx, (forecast, ts) in enumerate(zip(self.forecasts, self.tss)):
            
            if self.plot_only:
                ax = plt.subplot(1,1, 1)  # Change the grid size to 5x5
                if  self.plot_only in forecast.item_id:
                    zoom = -2 if zoom_to_predicted else 2
                    plt.plot(ts[zoom * self.prediction_length:].to_timestamp(), label="target")
                    forecast.plot(color='g', show_label=True)
                    plt.xticks(rotation=60)
                    ax.xaxis.set_major_formatter(date_formater)
                    ax.set_title(self.title+': '+forecast.item_id)
            else:
                ax = plt.subplot(6,4, idx + 1)  # Change the grid size to 5x5
                plt.plot(ts[-4 * self.prediction_length:].to_timestamp(), label="target")
                forecast.plot(color='g', show_label=True)
                plt.xticks(rotation=60)
                ax.xaxis.set_major_formatter(date_formater)
                ax.set_title(self.title+': '+forecast.item_id)
        
        plt.gcf().tight_layout()
        plt.legend()
        # Save the figure if a save path is provided
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def evaluate(self):
        evaluator = Evaluator()
        agg_metrics, ts_metrics = evaluator(iter(self.tss), iter(self.forecasts))
        return agg_metrics, ts_metrics
    

class plots_and_evaluation_2:
    
    def __init__(self, forecasts, tss, forecasts_2, tss_2, prediction_length=11, title='', plot_only=None):
        self.forecasts = forecasts
        self.tss = tss
        self.forecasts_2 = forecasts_2
        self.tss_2 = tss_2
        self.prediction_length = prediction_length
        self.title = title
        self.plot_only = plot_only
    
    def plot_forcasts_all(self, Dataset_name,zoom_to_predicted=False, save_path=None):
        plt.figure(figsize=(10,7) if self.plot_only else (20, 27))
        date_formater = mdates.DateFormatter('%Y,%m')
        plt.rcParams.update({'font.size': 9})

        for idx, ((forecast, ts), (forecast_2, ts_2)) in enumerate(zip(zip(self.forecasts, self.tss), zip(self.forecasts_2, self.tss_2))):
            if self.plot_only:
                ax = plt.subplot(1, 1, 1)
                if self.plot_only in forecast.item_id:
                    zoom = -5 if zoom_to_predicted else 20
                    plt.plot(ts[zoom * self.prediction_length:].to_timestamp(), label=f"Target: {Dataset_name} - { self.plot_only}")
                    forecast.plot(ax=ax, color='green', name=f"Fine-Tuned", show_label=True)
                    # plt.plot(ts_2[zoom * self.prediction_length:].to_timestamp(), label="Target 2", color='orange')
                    forecast_2.plot(ax=ax, color='red', name="Baseline", show_label=True)
                    plt.xticks(rotation=60)
                    ax.xaxis.set_major_formatter(date_formater)
                    ax.set_title(self.title + ': ' + forecast.item_id)
                    ax.set_xlabel('Date', fontsize=12)  # Set x-axis label with a specific size
                    ax.set_ylabel('Value', fontsize=12)  # Set y-axis label with a specific size
                    ax.set_title(self.title + ': ' + forecast.item_id, fontsize=14)  # Set title with a specific size
                    ax.tick_params(axis='x', labelsize=10)  # Adjust x-axis tick label sizes
                    ax.tick_params(axis='y', labelsize=10)  # Adjust y-axis tick label sizes

            else:
                ax = plt.subplot(6, 4, idx + 1)
                plt.plot(ts[-4 * self.prediction_length:].to_timestamp(), label="Target", color='blue')
                forecast.plot(ax=ax, color='green', name="Fine-Tuned", show_label=True)
                # plt.plot(ts_2[-4 * self.prediction_length:].to_timestamp(), label="Target 2", color='orange')
                forecast_2.plot(ax=ax, color='red', name="Baseline", show_label=True)
                plt.xticks(rotation=60)
                ax.xaxis.set_major_formatter(date_formater)
                ax.set_title(forecast.item_id)

        plt.gcf().tight_layout()
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def evaluate(self):
        evaluator = Evaluator()
        agg_metrics, ts_metrics = evaluator(iter(self.tss), iter(self.forecasts))
        agg_metrics_2, ts_metrics_2 = evaluator(iter(self.tss_2), iter(self.forecasts_2))
        return agg_metrics, ts_metrics, agg_metrics_2, ts_metrics_2
    
class EvaluationResults:
    def __init__(self, dataset_name, context_factor, model_type, overall_mape,overall_mase, all_industries_mape, forecasts, ts):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.overall_mape = overall_mape
        self.all_industries_mape = all_industries_mape
        self.forecasts = forecasts
        self.ts = ts
        self.context_factor = context_factor
        self.overall_mase = overall_mase

    def to_dict(self, feature_indices=None):
        result_dict = {
            "Dataset": self.dataset_name,
            'context_factor': self.context_factor,
            "Model Type": self.model_type,
            "Overall MAPE": self.overall_mape,
            "Overall MASE": self.overall_mase,
            "All Industries MAPE": self.all_industries_mape
        }
        if feature_indices is not None:
            result_dict['Feature Indices'] = feature_indices
        return result_dict


# Setup logging
logger = logging.getLogger('ModelEvaluationLogger')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('model_evaluation.log', maxBytes=100000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def evaluate_model(model_name, dataset_name, path_to_csv, model_type, batch_size=64, prediction_length=11, context_length_factor=6, max_epochs=100, with_external=False, initial_weights_path=r"lag-llama.ckpt", selected_features=None):
    try:
        logger.info(f"Starting evaluation of {model_name} on {dataset_name}")
        
        if model_name == 'lag-llama':
            llama_models_obj = llama_models(path_to_csv=path_to_csv, batch_size=batch_size, prediction_length=prediction_length, context_length_factor=context_length_factor, with_external=with_external, selected_features=selected_features)
            if model_type == 'baseline':
                forecasts, ts = llama_models_obj.baseline_llama_on_csv()
                logger.debug("Loaded baseline llama model forecasts.")
            elif model_type == 'fine_tuned':
                forecasts, ts = llama_models_obj.fine_tuned_model(max_epochs=max_epochs, initial_weights_path=initial_weights_path)
                logger.debug("Loaded fine-tuned llama model forecasts.")
            else:
                logger.error("Invalid model type provided.")
                raise ValueError("Invalid model type. Choose 'baseline' or 'fine_tuned'.")
        
        elif model_name == 'FeedForwardNetwork':
            forecasts, ts = run_feed_forward_model(path_to_csv=path_to_csv, prediction_length=prediction_length, context_length_factor=context_length_factor, with_external=with_external, selected_features=selected_features)
            logger.debug("Loaded FeedForwardNetwork forecasts.")
        
        elif model_name == 'tft':
            forecasts, ts = run_tft(path_to_csv=path_to_csv, prediction_length=prediction_length, context_length_factor=context_length_factor, with_external=with_external, selected_features=selected_features)
            logger.debug("Loaded TFT forecasts.")
        else:
            logger.error("Model name not recognized.")
            raise ValueError("Invalid model name provided. Please provide a valid model name.")
        
        agg_metrics, ts_metrics = plots_and_evaluation(forecasts, ts).evaluate()
        overall_mape = agg_metrics['MAPE']
        overall_mase = agg_metrics['MASE']
        logger.info(f"Calculated aggregate metrics for {model_name}.")

        # Handling specific dataset names
        if 'job' in dataset_name:
            column = dataset_name[:3] + '_VALUE_' + 'Total, all industries' + '_Job vacancies' if with_external else 'Total, all industries' #original 
            column_2024 = 'Total, all industries'
            all_industries_mape = ts_metrics[ts_metrics['item_id'] == column_2024]['MAPE'].item()
        elif dataset_name == 'emp_h':
            all_industries_mape = ts_metrics['MAPE'].item()
        else:
            column = dataset_name[:3] + '_VALUE_' + 'Industrial aggregate excluding unclassified businesses [11-91N]' if with_external else 'Industrial aggregate excluding unclassified businesses [11-91N]'
            all_industries_mape = ts_metrics[ts_metrics['item_id'] == column]['MAPE'].item()

        logger.info(f"Model evaluation completed for {model_name} on {dataset_name}.")
        return EvaluationResults(dataset_name, context_length_factor, model_type, overall_mape, overall_mase, all_industries_mape, forecasts, ts)
    
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}", exc_info=True)
        raise e

