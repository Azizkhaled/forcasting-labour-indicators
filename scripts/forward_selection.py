import os
import pandas as pd
from tqdm import tqdm
from scripts.plots_and_evaluation import evaluate_model

def forward_selection(df, plots_dir, model_name, datasets, feature_prefixes):
    """
    Perform forward feature selection to optimize model performance.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - plots_dir: Directory where results will be saved.
    - model_name: Name of the model used for evaluation.
    - datasets: List of tuples where each tuple contains:
        (dataset_name, path_to_csv, context_length_factor).
    - feature_prefixes: List of prefixes for grouping features.

    Returns:
    - final_selected_features: List of features selected after the forward selection process.
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
    good_features_groups_idxs = []
    best_score = float('inf')  # Start with infinity, the lower MAPE the better
    selected_features = []  # Start with no additional features

    # Establish baseline performance with no additional features
    baseline_result = evaluate_model(model_name, datasets[0][0], datasets[0][1], 'baseline',
                                     with_external=True, context_length_factor=datasets[0][2])
    baseline_score = baseline_result.overall_mape
    results.append(baseline_result.to_dict())

    # Forward selection process
    for dataset_name, path_to_csv, cl in tqdm(datasets, desc="Processing datasets"):
        for i, feature_group in tqdm(enumerate(feature_groups), desc="Forward feature selection"):
            current_features = selected_features + feature_group  # Add new group to selected features
            current_feature_indices = good_features_groups_idxs + [i]  # Include all previous good indices and the current index

            print(f'Processing {dataset_name} with feature group {i}')
            
            # Evaluate with the current set of selected features
            result = evaluate_model(model_name, dataset_name, path_to_csv, 'baseline',
                                    with_external=True, context_length_factor=cl,
                                    selected_features=current_features)
            new_score = result.overall_mape
            if new_score < best_score:
                best_score = new_score
                good_features_groups_idxs = current_feature_indices[:]  # Copy the current indices
                print(f'New best score: {best_score}, with feature groups: {good_features_groups_idxs}')
                selected_features = current_features  # Update the selected features

            # Convert the result to a dictionary and include the feature indices
            result_dict = result.to_dict(feature_indices=current_feature_indices)
            results.append(result_dict)

            # Save results to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(plots_dir, f"{plots_dir}_evaluation_results.csv"), index=False)

    # Compile final selected features based on selected feature indices
    final_selected_features = []
    for idx in good_features_groups_idxs:
        final_selected_features.extend(feature_groups[idx])  # Cumulatively add feature groups

    print("Final selected features:", final_selected_features)

    return final_selected_features
