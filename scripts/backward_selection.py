import os
import pandas as pd
from tqdm import tqdm
from scripts.plots_and_evaluation import evaluate_model

def backward_selection(df, plots_dir, model_name, dataset_name, path_to_csv, cl, feature_prefixes):
    """
    Perform backward feature selection to optimize model performance.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - plots_dir: Directory where results will be saved.
    - model_name: Name of the model used for evaluation.
    - datasets: List of tuples where each tuple contains:
        (dataset_name, path_to_csv, context_length_factor).
    - feature_prefixes: List of prefixes for grouping features.

    Returns:
    - final_selected_features: List of features selected after the backward selection process.
    """
    # Ensure the results directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Define feature groups based on prefixes
    feature_groups = [
        [col for col in df.columns if col.startswith(prefix)] 
        for prefix in feature_prefixes
    ]

    # Initialize variables
    results = []
    remaining_feature_groups_idxs = list(range(len(feature_groups)))  # Start with all feature groups
    best_score = float('inf')

    # Baseline performance with all features
    all_features = [feature for group in feature_groups for feature in group]
    baseline_result = evaluate_model(model_name, dataset_name, path_to_csv, 'fine_tuned',
                                     with_external=True, context_length_factor=cl, selected_features=all_features)
    baseline_score = baseline_result.all_industries_mape
    results.append(baseline_result.to_dict())

    # Backward selection process
    while remaining_feature_groups_idxs:
        score_improved = False
        for i in list(remaining_feature_groups_idxs):  # Iterate over a copy since we might modify the list
            # Try model without the current feature group
            trial_feature_groups_idxs = [idx for idx in remaining_feature_groups_idxs if idx != i]
            trial_features = [feature for idx in trial_feature_groups_idxs for feature in feature_groups[idx]]

            print(f'Trying without feature group {i}...')
            trial_result = evaluate_model(model_name, dataset_name, path_to_csv, 'fine_tuned',
                                            with_external=True, context_length_factor=cl,
                                            selected_features=trial_features)
            trial_score = trial_result.all_industries_mape

            # Convert the result to a dictionary and include the trial feature indices
            result_dict = trial_result.to_dict(feature_indices=trial_feature_groups_idxs)
            results.append(result_dict)
            trial_score_value = trial_score.iloc[0] if isinstance(trial_score, pd.Series) else trial_score
            print("trial score: ",trial_score_value,"##### best_score: ", best_score)
            # If performance improved, update best score and remove the feature group
            if trial_score_value < best_score:
                best_score = trial_score_value
                remaining_feature_groups_idxs = trial_feature_groups_idxs  # Update remaining groups
                score_improved = True
                print(f'Removing feature group {i} improved score to {best_score}')

            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(plots_dir, f"{plots_dir}_evaluation_results_bs.csv"), index=False)

        if not score_improved:
            print("No improvement from removing any more features.")
            break

    # Compile final selected features based on remaining indices
    final_selected_features = [feature for idx in remaining_feature_groups_idxs for feature in feature_groups[idx]]
    print("Final selected features:", final_selected_features)

    return final_selected_features,best_score
