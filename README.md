# FuDiCo

Repository for **FuDiCo: Gene Fusion-Initiated Path Propagation for Disease Comorbidity Prediction**.

## Before Running FuDiCo

### 1. Configure the Project Root

Set the project root directory in `main_config.py`.

This directory will be used to store all data files and results.

### 2. Install the Environment

Create and activate the Conda environment using the provided environment file:

```bash
conda env create -f FuDiCo_env.yml
conda activate FuDiCo_env
```
### 3. Download the Processed Data

Download the processed data folder **[`processed_data`](https://www.dropbox.com/scl/fo/8umsd9el3lzhdg0gbmw4v/AHqV0zYwV1J1iSDTJdXUg04?rlkey=pnjknapnftl4tl6kybk69acfj&st=prze81jo&dl=0)** and place it in `Data_Results/Data/`

## Precompute Diffusion Operators and Fusion-to-Subgraph Paths

Run the preprocessing script to generate diffusion operators and fusion-to-subgraph paths:

```bash
python precompute_graph_diffusion_and_paths.py
```

## Precomputed Data

All diffusion operators and fusion-to-subgraph paths have already been precomputed and are available for download.

Download the precomputed data folder **[`fusion_diffusion_data`](https://www.dropbox.com/scl/fo/yp4u8d3mhy0xbpww3fxan/AJtwx5iY2S_Ka_NSxuHXSBc?rlkey=qvf7aci6db1i73c07plrtdjd1&st=8o1u508d&dl=0)** and place it in: `Data_Results/Data/`

**Note:** The provided `fusion_diffusion_data` folder already contains the precomputed diffusion reachability matrices, fusion gene sets, and fusion-to-connected-component (CC) paths required for both training and testing FuDiCo. If these files are present, FuDiCo will automatically load and reuse them instead of recomputing them.

## Training FuDiCo

Train FuDiCo using: 

```bash
python train_FuDiCo.py -c FuDiCo_train_hyperparameters.json
```

## Testing FuDiCo

Before testing, download the model checkpoint file **[`FuDiCo_test_model.ckpt`](https://www.dropbox.com/scl/fi/8ean1oxaxx9texq8tvtir/FuDiCo_test_model.ckpt?rlkey=g7r8ppwdzq5qcffyeco6o0tgq&st=1mxz4ez7&dl=0)** and place it in `Data_Results/Results/Test_Resources/` alongside the testing hyperparameters file `FuDiCo_test_hyperparameters.json`.

Test FuDiCo using:

```bash
python test_FuDiCo.py --model_file FuDiCo_test_model.ckpt --test_config_file FuDiCo_test_hyperparameters.json
```

