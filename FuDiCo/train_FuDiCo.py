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

# Pytorch Lightning Imports
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Ours
sys.path.insert(0, '..') 
import FuDiCo as mdl
import main_config as config



def get_config_path():
    """
    Parse the command-line argument for the configuration file path.
    """
    
    argument_parser = argparse.ArgumentParser(description="Specify configuration file for training FuDiCo.")
    argument_parser.add_argument('--train_config_file', '-c', type=str, help='Path to the training configuration file.', required=True)
    parsed_args = argument_parser.parse_args()
    
    return parsed_args


def load_config_file(file_path):
    """
    Load and parse a JSON training configuration file.
    """
    
    with open(file_path, 'rt') as file_handle:
        config_data = commentjson.load(file_handle, object_hook=OrderedDict)
        
    return config_data


def get_hyperparameters(train_config):  
    """
    Retrieve FuDiCo hyperparameters from the training configuration.
    """
    
    hyperparameters = dict(train_config["FuDiCo_hyperparameters"])
    
    return hyperparameters    


def build_fudico_model(train_config):
    """
    Create FuDiCo model using the specified configuration.
    
    Returns:
        - model (FuDiCo): Initialized FuDiCo model with loaded configuration.
    """

    # Get FuDiCo hyperparameters 
    hyperparameters = get_hyperparameters(train_config)
    
    # Set random seeds for reproducibility
    random.seed(hyperparameters['seed'])
    np.random.seed(hyperparameters['seed'])
    torch.manual_seed(hyperparameters['seed'])
    torch.cuda.manual_seed(hyperparameters['seed'])
    torch.cuda.manual_seed_all(hyperparameters['seed']) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize FuDiCo model with paths and hyperparameters
    model = mdl.FuDiCo(
        hyperparameters,                                        # Model hyperparameters
        train_config["PPI_network_path"],                       # Path to PPI network 
        train_config["disease_pairs_path"],                     # Path to disease pairs 
        train_config["subgraphs_path"],                         # Path to disease subgraphs
        train_config["node_embeddings_path"],                   # Path to pretrained node embeddings
        train_config["fusion_pairs_path"],                      # Path to fusion gene pairs 
        train_config["fusion_diffusion_data_path"]              # Path to fusion diffusion data
                      ) 
    
    return model, hyperparameters


def build_fudico_trainer(train_config, hyperparameters):
    """
    Create a PyTorch Lightning Trainer with TensorBoard logging.
    """

    # Set up TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=train_config['tensorboard_dir'],
        name=None,
        version=("version_" + str(random.Random().randint(0, 1000000000))))
    if not os.path.exists(logger.log_dir):
        os.makedirs(logger.log_dir, exist_ok=True)
   
    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir),
        filename="{epoch}-{val_average_precision:.2f}-{val_accuracy:.2f}-{val_auroc:.2f}-{val_f1:.2f}",
        save_top_k=-1, 
        verbose=True,
        monitor=hyperparameters['monitor_metric'],
        mode='max')
 
    # Trainer configuration
    trainer_kwargs = {
        'max_epochs': hyperparameters['max_epochs'],
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1, 
        "num_sanity_val_steps": 0,
        "gradient_clip_val": hyperparameters['grad_clip'],
        "enable_progress_bar": True,
        "callbacks": [checkpoint_callback],
        "logger": logger,
                      }
    
    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    return trainer, trainer_kwargs, logger.log_dir


def train_fudico_model(train_config):  
    """
    Train, validate, and test the FuDiCo model.
    """

    # Build the model and retrieve the hyperparameters
    model, hyperparameters = build_fudico_model(train_config)

    # Initialize the trainer and logging configuration
    trainer, trainer_kwargs, results_path = build_fudico_trainer(train_config, hyperparameters)

    # Save hyperparameters to the results directory
    with open(os.path.join(results_path, "FuDiCo_test_hyperparameters.json"), "w") as hyperparameters_file:
        json.dump({"FuDiCo_hyperparameters": hyperparameters}, hyperparameters_file, indent=4)

    # Save trainer configurations to the results directory
    with open(os.path.join(results_path, "trainer_config.json"), "w") as tkwarg_file:
            pop_keys = [key for key in ['logger','profiler','early_stop_callback','callbacks'] if key in trainer_kwargs.keys()]
            for key in pop_keys:
                trainer_kwargs.pop(key, None) 
            tkwarg_file.write(json.dumps(trainer_kwargs, indent=4))
            
    # Start training
    print("...Start Training FuDiCo Model...")
    trainer.fit(model)
    print("\n\n...Finished Training FuDiCo Model...", flush=True)

    # Compute the best validation metric
    best_val_metrics = max(model.val_metric_scores, key=lambda x: x[hyperparameters['monitor_metric']])
    print("\n...Best Model Performance on Validation Set...")
    print(f"Achieved at Epoch {best_val_metrics['epoch']}:")
    for metric, value in best_val_metrics.items():
        if metric != 'epoch':
            print(f"{metric.replace('_', ' ').capitalize()}: {value:.4f}")

    # Test the model
    print("\n...Testing the Best Model (Once after all training epochs)...")
    trainer.test(model)
    
    # Save test results to file
    test_results = model.test_epoch_metrics
    filtered_test_metrics = {
        'test_loss': test_results['test_loss'],
        'test_accuracy': test_results['test_accuracy'],
        'test_f1': test_results['test_f1'],
        'test_average_precision': test_results['test_average_precision'],
        'test_auroc': test_results['test_auroc']}
    with open(os.path.join(results_path, "test_results.json"), "w") as test_result_file:
            json.dump(filtered_test_metrics, test_result_file, indent=4)
    print("\n\n...Finished Testing the Model and Test Results Saved...", flush=True)

    return best_val_metrics

 
def main():
    """
    Run FuDiCo training, validation, and testing pipeline using the provided configuration file.
    """

    config_args  = get_config_path()
    
    # Set up base directories for data and results
    base_data_path = Path(config.PROJECT_ROOT) / "Data" / "processed_data"
    base_result_path = Path(config.PROJECT_ROOT) / "Results"
    base_result_path.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = base_result_path / "Train_Results" / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    test_resources_dir = base_result_path / "Test_Resources"
    test_resources_dir.mkdir(parents=True, exist_ok=True)

    # Load training configuration from JSON file
    train_config = load_config_file(config_args.train_config_file)

    # Assign paths for input data and precomputed fusion diffusion data
    train_config['tensorboard_dir'] = str(tensorboard_dir)
    train_config["PPI_network_path"] = os.path.join(base_data_path, "PPI.txt")
    train_config["disease_pairs_path"] = os.path.join(base_data_path, "disease_pairs.txt")
    train_config["subgraphs_path"] = os.path.join(base_data_path, "disease_subgraphs.pth")
    train_config["node_embeddings_path"] = os.path.join(base_data_path, "ESM-2_gene_embeddings.pth")
    train_config["fusion_pairs_path"] = os.path.join(base_data_path, "fusion_gene_pairs.txt")
    train_config["fusion_diffusion_data_path"] = os.path.join(base_data_path, "fusion_diffusion_data")

    train_fudico_model(train_config)

        
if __name__ == "__main__":
    main()
