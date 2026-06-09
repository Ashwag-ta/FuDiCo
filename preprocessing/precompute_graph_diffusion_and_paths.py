# General Imports
from collections import defaultdict
from itertools import chain
import multiprocessing 
import numpy as np

# Networkx Import
import networkx as nx

# Preprocessing Configuration
import prep_config as config



def nested_defaultdict_list():
    
    return defaultdict(list)


def extract_paths_for_fusion_node(args):
    
    idx, fusion_node, graph, subgraph_node_ids, max_path_length, max_paths_per_pair, total_fus_nodes = args
    paths_by_k = defaultdict(nested_defaultdict_list)
    
    print(f"...Processing {idx + 1}/{total_fus_nodes} fusion node {fusion_node}...")
    target_nodes = set(subgraph_node_ids) - {fusion_node}
    all_paths = nx.all_simple_paths(graph, source=fusion_node, target=target_nodes, cutoff=max_path_length)
    
    for path in all_paths:
        k = len(path) - 1  # path length
        target_node = path[-1]
        stored_paths = paths_by_k[k][(fusion_node, target_node)]
        if len(stored_paths) < max_paths_per_pair:
            stored_paths.append(path)

    print(f"...Finished processing fusion node {fusion_node}...")

    return paths_by_k


def precompute_graph_diffusion_and_paths(PPI_graph, PPI_node_ids, fusion_node_ids, subgraph_node_ids, num_processes, max_path_length, max_paths_per_pair):
    """
    Precompute diffusion operators and fusion-to-subgraph paths. 
    
    Args:
        - PPI_graph (nx.Graph): The full protein–protein interaction (PPI) graph.
        - PPI_node_ids (List[int]): Ordered list of PPI node IDs.
        - fusion_node_ids (List[int]): Fusion-associated gene IDs that act as source nodes for path extraction. 
        - subgraph_node_ids (List[int]): All nodes belonging to disease-associated subgraphs acting as targets.
        - max_path_length (int): Maximum path length used for diffusion operator and extracting simple paths.
        - max_paths_per_pair (int): Maximum number of paths retained for each (fusion node, target node) pair.

    Saves:
        - diffusion_operator_k : Diffusion operator powers for k = 1..max_path_length
        - paths_by_k : Fusion-to-subgraph paths grouped by length (k)
    """
    
    diffusion_paths_dir = config.DATA_RESULTS_DIR / "Data" / "fusion_diffusion_data"
    diffusion_paths_dir.mkdir(parents=True, exist_ok=True)

    # --- Precompute powers of the symmetric normalized diffusion operator up to max_path_length --- #
    if config.PRECOMPUTE_DIFFUSION_OPERATORS:
        
        # Compute the symmetric degree-normalized diffusion operator
        norm_laplacian = nx.normalized_laplacian_matrix(PPI_graph, nodelist=PPI_node_ids, weight=None).toarray().astype(np.float32)  # Compute normalized graph Laplacian
        sym_norm_diff_op = np.eye(norm_laplacian.shape[0], dtype=np.float32) - norm_laplacian
        
        # Precompute powers of the diffusion operator 
        for k in range(1, max_path_length + 1):
            diffusion_operator_k_path = diffusion_paths_dir / f'diffusion_operator_k_{k}.npy'
            if not diffusion_operator_k_path.exists():
                print(f"...Computing diffusion operator (power={k})...")
                diffusion_operator_k = np.linalg.matrix_power(sym_norm_diff_op, k)
                np.save(diffusion_operator_k_path, diffusion_operator_k)
                print(f"...Saved diffusion operator (power={k})...")
            else:
                print(f"...Diffusion operator already exists (power={k})...")
  
    # --- Precompute fusion-to-subgraph simple paths up to max_path_length --- #
    if config.PRECOMPUTE_FUSION_SUBGRAPH_PATHS:
        
        all_paths_k_exist = True
        
        for k in range(1, max_path_length + 1):
            fusion_sub_paths_k_path = diffusion_paths_dir / f"fusion_to_subgraph_paths_len_{k}.npy"
            if not fusion_sub_paths_k_path.exists():
                all_paths_k_exist = False
                break

        if all_paths_k_exist:
            print("...Fusion-to-subgraph paths already exist...")
        else:
            print("...Precomputing fusion-to-subgraph paths...")
            tasks = [(fus_idx, fusion_node, PPI_graph, subgraph_node_ids, max_path_length, max_paths_per_pair, len(fusion_node_ids))
                for fus_idx, fusion_node in enumerate(fusion_node_ids)
                    ]

            # Initialize dictionary of paths grouped by k
            paths_by_k = {k: defaultdict(list) for k in range(1, max_path_length + 1)}

            with multiprocessing.Pool(processes=num_processes) as pool:
                for paths_by_k_chunk in pool.imap_unordered(extract_paths_for_fusion_node, tasks):
                    for k, paths_by_pair in paths_by_k_chunk.items():
                        for fusion_target_pair, paths in paths_by_pair.items():
                            paths_by_k[k][fusion_target_pair].extend(paths)

            # Save fusion-to-subgraph paths grouped by k
            for k, fusion_target_paths in paths_by_k.items():
                paths_k_path = diffusion_paths_dir / f"fusion_to_subgraph_paths_len_{k}.npy"
                np.save(paths_k_path, fusion_target_paths, allow_pickle=True)
                print(f"...Saved fusion-to-subgraph paths (length={k})...")


def main():
    
    # Load the full PPI graph
    PPI_graph = nx.relabel_nodes(nx.read_edgelist(str(config.DATA_RESULTS_DIR / "Data" / "PPI.txt")), int)
    PPI_node_ids = sorted(PPI_graph.nodes())
 
    # Load fusion gene pairs as a graph and extract fusion-associated genes
    fusion_graph = nx.relabel_nodes(nx.read_edgelist(str(config.DATA_RESULTS_DIR / "Data" / "fusion_gene_pairs.txt")), int)
    fusion_node_ids = sorted(fusion_graph.nodes())

    # Load disease-associated genes for each subgraph 
    subgraphs_file = config.DATA_RESULTS_DIR / "Data" / "disease_subgraphs.pth"
    all_subgraphs = []
    with open(subgraphs_file) as sub_f:
        for idx, line in enumerate(sub_f):
            subgraph_nodes = [int(node) for node in line.split("\t")[0].split("-")]
            all_subgraphs.append(subgraph_nodes)
    subgraph_node_ids = sorted(set(chain.from_iterable(all_subgraphs)))
   
    precompute_graph_diffusion_and_paths(PPI_graph, PPI_node_ids, fusion_node_ids, subgraph_node_ids, config.N_PROCESSES, config.MAX_PATH_LENGTH, config.MAX_PATHS_PER_PAIR)


if __name__ == '__main__':
    main()
