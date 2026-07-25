# General Imports
import os
import sys
from collections import OrderedDict
import random
import argparse
import commentjson
import numpy as np
import json
from pathlib import Path

# PyTorch Import
import torch

# Matplotlib Imports
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Sci-kit Learn Import
from sklearn.metrics import RocCurveDisplay

# Pytorch Lightning Import
import lightning.pytorch as pl

# Ours
sys.path.insert(0, '..')
import FuDiCo as mdl
import main_config as config



def parse_arguments():
    """
    Parse the command-line arguments for specifying the model checkpoint and test configuration file.
    """
    
    argument_parser = argparse.ArgumentParser(description="Test a pre-trained FuDiCo model using multiple seeds.")
    argument_parser.add_argument('--model_file', type=str, default="FuDiCo_test_model.ckpt",
                        help="Filename of the trained model checkpoint (default: FuDiCo_test_model.ckpt).")
    argument_parser.add_argument('--test_config_file', type=str, default="FuDiCo_test_hyperparameters.json",
                        help="Filename of the test configuration file (default: FuDiCo_test_hyperparameters.json).")
    parsed_args = argument_parser.parse_args()
    
    return parsed_args

    
def load_config_file(file_path):
    """
    Load and parse a JSON testing configuration file.
    """
    
    with open(file_path, 'rt') as file_handle:
        test_hyperparameters = commentjson.load(file_handle, object_hook=OrderedDict)
        
    return test_hyperparameters


def build_fudico_model(hyperparameters, test_config):
    """
    Create FuDiCo model using the specified configuration.
    
    Returns:
         - model (FuDiCo): Initialized FuDiCo model with loaded configuration.
    """
    
    model = mdl.FuDiCo(
        hyperparameters,                                        # Model hyperparameters
        test_config["PPI_network_path"],                        # Path to PPI network 
        test_config["disease_pairs_path"],                      # Path to disease pairs 
        test_config["subgraphs_path"],                          # Path to disease subgraphs
        test_config["node_embeddings_path"],                    # Path to pretrained node embeddings
        test_config["fusion_pairs_path"],                       # Path to fusion gene pairs 
        test_config["fusion_diffusion_data_path"]               # Path to fusion diffusion data
                      ) 
 
    return model


def load_pretrained_weights(model, model_path):
    """
    Load pre-trained weights into the given model from a checkpoint file.
    
    Returns:
        The model with loaded pre-trained weights.
    """
    
    print(f"...Loading checkpoint from: {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model.load_state_dict(pretrain_dict)
    
    return model


def test_fudico_model(hyperparameters, model_path, test_config, test_results_path):
    """
    Run the testing pipeline using a pre-trained model over multiple random seeds.
    """

    # Initialize metric storage for all seed runs
    all_run_results = {
        'accuracy': [],
        'f1': [],
        'average_precision': [],
        'auroc': []}

    # ROC curve data across all seed runs
    all_fpr = []  
    all_tpr = []  
    all_fpr_interpolated = np.linspace(0, 1, 100) 
    all_tpr_interpolated = [] 

    # PR curve data across all seed runs
    all_precision_curves = []
    all_recall_curves = []
    all_recalls_interpolated = np.linspace(0, 1, 100)

    # Loop through multiple seed runs  
    for seed in range(10):
        print(f"\n...Testing the Model for Seed Run {seed + 1}...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
        # Build and initialize the FuDiCo model
        model = build_fudico_model(hyperparameters, test_config)

        # Set up PyTorch Lightning trainer
        trainer = pl.Trainer(accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=1)

        # Load pre-trained model weights
        model = load_pretrained_weights(model, model_path)

        # Run model testing
        trainer.test(model)  
        
        # Retrieve stored test results from the model
        curr_seed_results = model.test_epoch_metrics 
        
        # Log the metrics from current seed run
        for metric in all_run_results.keys():
            all_run_results[metric].append(curr_seed_results[f'test_{metric}'])

        # Extract predicted scores and labels for ROC/PR curves
        pred_scores = np.array(curr_seed_results.get('pred_scores').cpu())
        labels = np.array(curr_seed_results.get('labels').cpu())
    
        # Process ROC curve for the current run
        fpr = np.array(curr_seed_results['fpr'])
        tpr = np.array(curr_seed_results['tpr'])
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        tpr_interp = np.interp(all_fpr_interpolated, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        all_tpr_interpolated.append(tpr_interp)
        RocCurveDisplay.from_predictions(labels, pred_scores, name=f"Run {seed + 1}")
        plt.title(f"ROC Curve for Run {seed + 1}")
        plt.savefig(os.path.join(test_results_path, f"roc_curve_Run_{seed + 1}.png"), dpi=600)
        plt.close()

        # Process PR curve for the current seed run
        precision_curve = np.array(curr_seed_results['precision_curve'])
        recall_curve = np.array(curr_seed_results['recall_curve'])
        all_precision_curves.append(precision_curve)
        all_recall_curves.append(recall_curve)
        plt.plot(recall_curve, precision_curve, label=f"Run {seed + 1} (AUPRC = {np.trunc(curr_seed_results['test_average_precision'] * 100) / 100.0})", color='green', linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve for Run {seed + 1}")
        plt.legend(loc="lower left", fancybox=True, fontsize=10)  
        plt.savefig(os.path.join(test_results_path, f"pr_curve_Run_{seed + 1}.png"), dpi=600 , bbox_inches="tight")
        plt.close()
    
    # Compute average metrics across all seed runs
    summary_results = {}
    for metric in ['accuracy', 'f1', 'average_precision', 'auroc']:
        metric_values = all_run_results[metric]
        metric_mean = np.mean(metric_values)
        summary_results[f"{metric}_mean"] = metric_mean

    print("\n...Mean Testing Results Across All Seed Runs...")
    for metric, value in summary_results.items():
        formatted_metric = metric.replace("_", " ").title()
        print(f"{formatted_metric}: {value}")

    # Save the final results
    final_results = {
        "Individual Seed Run Results": {
            metric: [v for v in values] for metric, values in all_run_results.items()
        },
        "Mean Seed Run Results": {
            metric.replace("_", " ").title(): str(value) for metric, value in summary_results.items()
        }
    }
    
    with open(os.path.join(test_results_path, "final_test_results.json"), "w") as results_file:
        json.dump(final_results, results_file, indent=4)
    print("\n...All Individual and Mean Seed Run Results Saved...")

    
    # Plot all individual and mean ROC and PR curves across multiple seed runs
    custom_colors = ["#000080", "#d62728", "#228b22", "#ff4500", "#4B0082", "#4d4d4d", "#E1AD01", "#008b8b", "#FF00FF", "#40E0D0"]
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Individual ROC curves
    for i, (fpr, tpr, auroc) in enumerate(zip(all_fpr, all_tpr, all_run_results["auroc"])):
         ax_roc.plot(fpr, tpr, color=custom_colors[i % len(custom_colors)], alpha=0.6, linewidth=1.5, label=f"Run {i+1} (AUROC = {np.trunc(auroc * 100) / 100.0})")

    # Mean ROC curve
    mean_tpr = np.mean(all_tpr_interpolated, axis=0)
    ax_roc.plot(all_fpr_interpolated, mean_tpr, label=f"Mean ROC Curve (Mean AUROC = {np.trunc(summary_results['auroc_mean'] * 100) / 100.0})", color="#8b4513", linewidth=2)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right", fontsize=7, frameon=True)

    # Individual PR curves
    for i, (precision, recall, auprc) in enumerate(zip(all_precision_curves, all_recall_curves,  all_run_results["average_precision"])):
         ax_pr.plot(recall, precision, color=custom_colors[i % len(custom_colors)], alpha=0.6, linewidth=1.5, label=f"Run {i+1} (AUPRC = {np.trunc(auprc * 100) / 100.0})")

    # Mean PR curve
    all_precisions_interpolated = [
        np.interp(all_recalls_interpolated, recall[::-1], precision[::-1])
        for precision, recall in zip(all_precision_curves, all_recall_curves)]
    mean_precision = np.mean(all_precisions_interpolated, axis=0)
    ax_pr.plot(all_recalls_interpolated, mean_precision, label=f"Mean PR Curve (Mean AUPRC = {np.trunc(summary_results['average_precision_mean'] * 100) / 100.0})",
               color="#8b4513", linewidth=2)
    
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax_pr.legend(loc="lower left", fontsize=7, frameon=True)

    # Save the combined figure
    fig.savefig(os.path.join(test_results_path, "all_roc_pr_curves_combined.png"), dpi=600, bbox_inches="tight")

    plt.close(fig)
    print("\n...All ROC and Precision-Recall Curves (Individual and Mean) Saved...")
    print("\n...Finished Testing the Model...")

        
def main():
    """
    Run FuDiCo testing pipeline using the test hyperparameters and pre-trained model.
    """
    
    args = parse_arguments()

    # Set up base directories for data and results
    base_data_path = Path(config.PROJECT_ROOT) / "Data" / "processed_data"
    test_resources_path = Path(config.PROJECT_ROOT) / "Results" / "Test_Resources" 
    test_results_path = Path(config.PROJECT_ROOT) / "Results" / "Test_Results" 
    test_results_path.mkdir(parents=True, exist_ok=True)
    
    model_path = test_resources_path / args.model_file
    test_config_path = test_resources_path / args.test_config_file
    
    # Load the test configuration from JSON file
    test_config = load_config_file(test_config_path)

    # Extract model hyperparameters from the test configuration file
    hyperparameters = test_config["FuDiCo_hyperparameters"]

    # Assign paths for input data and precomputed fusion diffusion data
    test_config["PPI_network_path"] = os.path.join(base_data_path, "PPI.txt")
    test_config["disease_pairs_path"] = os.path.join(base_data_path, "disease_pairs.txt")
    test_config["subgraphs_path"] = os.path.join(base_data_path, "disease_subgraphs.pth")
    test_config["node_embeddings_path"] = os.path.join(base_data_path, "ESM-2_gene_embeddings.pth")
    test_config["fusion_pairs_path"] = os.path.join(base_data_path, "fusion_gene_pairs.txt")
    test_config["fusion_diffusion_data_path"] = os.path.join(config.PROJECT_ROOT, "Data", "fusion_diffusion_data")

    # Run the model testing
    test_fudico_model(hyperparameters, model_path, test_config, test_results_path)   


if __name__ == "__main__":
    main()
