# General Imports
import sys
from pathlib import Path

# Import project configuration
# Add parent directory to path to enable importing main config
sys.path.insert(0, '..')
import main_config as config



# Directory where precomputed data will be saved
DATA_RESULTS_DIR = config.PROJECT_ROOT

# Flags for precomputing diffusion operators and fusion-to-subgraph paths
MAX_PATH_LENGTH = 3 # Maximum path length (k)
MAX_PATHS_PER_PAIR = 100 # Maximum number of paths per (fusion, target) pair
PRECOMPUTE_DIFFUSION_OPERATORS = True # Precompute powers of the symmetric normalized diffusion operator
PRECOMPUTE_FUSION_SUBGRAPH_PATHS = True # Precompute fusion-to-subgraph paths
N_PROCESSES = 5  # Number of processes for parallel path extraction






