import pickle
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
from scripts.plots_and_evaluation import evaluate_model, plots_and_evaluation
from scripts.data_processing import data_creation

def final_model_main(df_path, remaining_feature_groups_idxs, feature_prefixes, plots_dir, model_name, dataset_creation, datasets):
    """
    Main function to evaluate models with selected features and save the results.

    Parameters:
    - df_path: Path to the main dataset CSV file.
    - remaining_feature_groups_idxs: List of indices for the remaining feature groups to include.
    - feature_prefixes: List of prefixes for grouping features.
    - plots_dir: Directory where plots and results will be saved.
    - model_name: Name of the model used for evaluation.
    - datasets: List of tuples where each tuple contains:
        (dataset_name, path_to_csv, context_length_factor).
    """
    # Create the dataset
    if dataset_creation:
        data_creation(time_steps = 0)

    # Load the dataset
    df = pd.read_csv(df_path)

    # Compile final selected features based on remaining indices
    feature_groups = [
        [col for col in df.columns if col.startswith(prefix)] 
        for prefix in feature_prefixes
    ]
    final_selected_features = [feature for idx in remaining_feature_groups_idxs for feature in feature_groups[idx]]

    # Ensure the results directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Evaluate models and save plots
    results = []
    best_score = float('inf')

    for dataset_name, path_to_csv, cl in tqdm(datasets, desc=f"Processing datasets"):
        best_score = float('inf')
        print(f'Processing {dataset_name} with {cl} context_length factor')
        for i in range(1):
            for model_type in ['fine_tuned']:
                result = evaluate_model(model_name, dataset_name, path_to_csv, model_type,
                                        prediction_length=12, # orignal 11
                                        with_external=True, context_length_factor=cl,
                                        selected_features=final_selected_features)

                results.append(result)
                if result.overall_mase < best_score:
                    best_score = result.overall_mase
                    best_result = result

                    # File path for the result pickle file
                    result_pickle_path = os.path.join(plots_dir, f'MASE_{result.dataset_name}_best_result.pkl')

                    # Check if the file exists and delete it if it does
                    if os.path.exists(result_pickle_path):
                        os.remove(result_pickle_path)

                    # Save the new best result
                    with open(result_pickle_path, 'wb') as f:
                        pickle.dump(best_result, f)

                # Save plots
                plot_obj = plots_and_evaluation(forecasts=result.forecasts, tss=result.ts, title=dataset_name + '_' + model_type)
                save_path = os.path.join(plots_dir, f"{i}_MASE_{result.dataset_name}_{cl}_forecasts.png")
                plot_obj.plot_forcasts_all(save_path=save_path)

                # Create DataFrame
                results_df = pd.DataFrame([result.to_dict() for result in results])

                # Save results to CSV
                results_df.to_csv(os.path.join(plots_dir, 'final_evaluation_results.csv'), index=False)
                # predictions_df = pd.DataFrame({
                #     'date': pd.date_range(start='2023-06-01', periods=12, freq='M'),
                #     'forecast': result.forecasts
                # })-
                # predictions_df.to_csv(os.path.join(plots_dir, f'{dataset_name}_2024_predictions.csv'), index=False)

if __name__ == "__main__":
    # Example parameters (you may customize these)
    df_path = 'datasets\emp_melt_complete_data.csv'
    remaining_feature_groups_idxs = [2, 4, 5, 6, 8, 9, 10]
    feature_prefixes = ['covid', 'indeed', 'inf_VALUE_Bank rate', 'inf_VALUE_Target rate',
                        'cpi_VALUE', 'gdp_VALUE', 'bus', 'job', 'ear', 'emp', 'hou']
    dataset_creation = True
    datetime_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    plots_dir = f"run-{datetime_stamp}"
    model_name = 'lag-llama'
    datasets = [
        # ('emp', 'datasets\emp_melt_complete_data.csv', 6), 
        #         ('earn', 'datasets\ear_melt_complete_data.csv', 6), 
        #         ('hours', 'datasets\hou_melt_complete_data.csv', 6), 
                # ('job', 'datasets\job_melt_complete_data_2024.csv', 5.2)
                ('emp_h', 'datasets\emp_health_melt_complete_data.csv', 6)
                ]

    # Call the main function
    final_model_main(df_path, remaining_feature_groups_idxs, feature_prefixes, plots_dir, model_name, dataset_creation, datasets)
    print(f"this run start on {datetime_stamp} and ended on {datetime.now()}")

