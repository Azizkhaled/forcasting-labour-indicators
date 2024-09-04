import pickle
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
from scripts.plots_and_evaluation import evaluate_model, plots_and_evaluation, plots_and_evaluation_2
from scripts.data_processing import data_creation
from scripts.backward_selection import backward_selection
from scripts.forward_selection import forward_selection

import pickle
import pandas as pd

import re

from scripts.plots_and_evaluation import evaluate_model, plots_and_evaluation, plots_and_evaluation_2

def final_model_main(df_path, remaining_feature_groups_idxs, feature_prefixes, plots_dir, model_name, dataset_creation,feat_backward_selection, feat_forward_selection, datasets):
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

    print(df)

    #run backwards selection
    if feat_backward_selection:
        final_selected_features = backward_selection(df, plots_dir, model_name, datasets, feature_prefixes)
        # feature_prefixes = {item.split('_', 1)[0] for item in selected_features}
    elif feat_forward_selection:
        final_selected_features = forward_selection(df, plots_dir, model_name, datasets, feature_prefixes)
        # feature_prefixes = {item.split('_', 1)[0] for item in selected_features}
    
    # Compile final selected features based on remaining indices
    feature_groups = [
        [col for col in df.columns if col.startswith(prefix)] 
        for prefix in feature_prefixes
    ]
    

    print(remaining_feature_groups_idxs)
    final_selected_features = [feature for idx in remaining_feature_groups_idxs for feature in feature_groups[idx]]

    # final_selected_features = [feature for sublist in feature_groups for feature in sublist]
    # final_selected_features =  [col for col in df.columns if col.startswith(feature_prefixes)]

    # Ensure the results directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Evaluate models and save plots
    results = []
    best_score = float('inf')

    for dataset_name, path_to_csv, cl in tqdm(datasets, desc=f"Processing datasets"):
        best_score = float('inf')
        model_type = 'fine_tuned'
        print(f'Processing {dataset_name} with {cl} context_length factor - {model_type}')
        
        for i in range(1):
                
                result = evaluate_model(model_name, dataset_name, path_to_csv, model_type,
                                        prediction_length=6, # orignal 11
                                        with_external=True, context_length_factor=cl,
                                        selected_features=final_selected_features)

                results.append(result)

                if result.overall_mase < best_score:
                    best_score = result.overall_mase
                    best_result = result

                    # File path for the result pickle file
                    result_pickle_path = os.path.join(plots_dir, f'{model_type}_{result.dataset_name}_best_result.pkl')

                    # Check if the file exists and delete it if it does
                    if os.path.exists(result_pickle_path):
                        os.remove(result_pickle_path)

                    # Save the new best result
                    with open(result_pickle_path, 'wb') as f:
                        pickle.dump(best_result, f)


                # Create DataFrame
                results_df = pd.DataFrame([result.to_dict() for result in results])

                # Save results to CSV
                results_df.to_csv(os.path.join(plots_dir, 'final_evaluation_results.csv'), index=False)
                print(feature_prefixes)
        
        best_score = float('inf')
        model_type = 'baseline'
        print(f'Processing {dataset_name} with {cl} context_length factor - {model_type}')
        
        for i in range(1):
                
                result = evaluate_model(model_name, dataset_name, path_to_csv, model_type,
                                        prediction_length=6, # original 11
                                        with_external=True, context_length_factor=cl,
                                        selected_features=final_selected_features)

                results.append(result)

                if result.overall_mase < best_score:
                    best_score = result.overall_mase
                    best_result = result

                    # File path for the result pickle file
                    result_pickle_path = os.path.join(plots_dir, f'{model_type}_{result.dataset_name}_best_result.pkl')

                    # Check if the file exists and delete it if it does
                    if os.path.exists(result_pickle_path):
                        os.remove(result_pickle_path)

                    # Save the new best result
                    with open(result_pickle_path, 'wb') as f:
                        pickle.dump(best_result, f)


                # Create DataFrame
                results_df = pd.DataFrame([result.to_dict() for result in results])

                # Save results to CSV
                results_df.to_csv(os.path.join(plots_dir, 'final_evaluation_results.csv'), index=False)
                print(feature_prefixes)     

        pickle_file_path_fine_tuned = f'{plots_dir}\\fine_tuned_{dataset_name}_best_result.pkl'
        pickle_file_path_baseline = f'{plots_dir}\\baseline_{dataset_name}_best_result.pkl'

        # plot_only = None
        # plot_only = 'Total, all industries' if dataset_name == 'job' else 'Industrial aggregate exclu'
        plot_only = 'health'

        # Load the best result from the pickle file
        with open(pickle_file_path_fine_tuned, 'rb') as f:
            loaded_best_result_fine_tuned = pickle.load(f)
        with open(pickle_file_path_baseline, 'rb') as f:
            loaded_best_result_baseline = pickle.load(f)
        plot_obj = plots_and_evaluation_2(forecasts= loaded_best_result_fine_tuned.forecasts, tss=loaded_best_result_fine_tuned.ts, forecasts_2=loaded_best_result_baseline.forecasts,
                                    tss_2=loaded_best_result_baseline.ts, title = loaded_best_result_fine_tuned.dataset_name, plot_only=plot_only)
        save_path_all = os.path.join(plots_dir, f"{loaded_best_result_fine_tuned.dataset_name}_both_forecasts_final_no_zoom.png")
        plot_obj.plot_forcasts_all(Dataset_name = loaded_best_result_fine_tuned.dataset_name, save_path=save_path_all, zoom_to_predicted= False)
        save_path_zoom = os.path.join(plots_dir, f"{loaded_best_result_fine_tuned.dataset_name}_both_forecasts_final_zoom.png")
        plot_obj.plot_forcasts_all(Dataset_name = loaded_best_result_fine_tuned.dataset_name, save_path=save_path_zoom, zoom_to_predicted= True)
                
if __name__ == "__main__":
    # Example parameters (you may customize these)
    # df_path = 'datasets\emp_melt_complete_data.csv'
    remaining_feature_groups_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # remaining_feature_groups_idxs = [1, 3, 4, 5]
    feature_prefixes = ['age_', 'pop_', 'covid_', 'indeed_', 'inf_',
                        'cpi_', 'gdp_', 'bus_', 'job_', 'ear_', 'emp_', 'hou_']
    dataset_creation = True
    feat_backward_selection = False
    feat_forward_selection= False
    datetime_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    
    model_name = 'lag-llama'
    datasets = [
                # ('emp', 'datasets\emp_melt_complete_data.csv', 12), 
                # ('earn', 'datasets\ear_melt_complete_data.csv', 6), 
                # ('hours', 'datasets\hou_melt_complete_data.csv', 6), 
                ('job', 'datasets\job_melt_complete_data.csv', 6),
                # ('emp_h', 'datasets\emp_health_melt_complete_data.csv', 6)
                ]
    
    plots_dir = f"run-{datetime_stamp}-{datasets[0][0]}-{'fs' if feat_forward_selection else 'bs' if feat_backward_selection else ''}"
    # Call the main function
    final_model_main(datasets[0][1], remaining_feature_groups_idxs, feature_prefixes, plots_dir, model_name, dataset_creation, feat_backward_selection, feat_forward_selection, datasets)
    print(f"this run start on {datetime_stamp} and ended on {datetime.now()}")

