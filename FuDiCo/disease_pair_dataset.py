# General Import
from typing import Dict

# PyTorch Imports
import torch
from torch.utils.data import Dataset



class DiseasePairDataset(Dataset): 
    """
    Dataset for disease pairs.
    """

    def __init__(self, disease_pairs, disease_pair_labels, cc_node_ids, fus_cc_paths: Dict, fus_cc_path_pos_influence_scores: Dict):

        # Model inputs
        self.disease_pairs = disease_pairs
        self.disease_pair_labels = disease_pair_labels
        self.cc_node_ids = cc_node_ids
        self.fus_cc_paths = fus_cc_paths
        self.fus_cc_path_pos_influence_scores = fus_cc_path_pos_influence_scores

        
    def __len__(self):
        """
        Return number of disease pairs.
        """
        
        return len(self.disease_pair_labels)


    def __getitem__(self, idx):
        """
        Retrieve one disease pair and its associated subgraph data.
        """

        # Get disease pair and label
        disease_pair = self.disease_pairs[:, idx]
        disease_pair_label = self.disease_pair_labels[idx]

        # Extract subgraph indices
        subgraph_a_idx = torch.tensor(disease_pair[0].item())
        subgraph_b_idx = torch.tensor(disease_pair[1].item())

        # Get CC node IDs
        subgraph_a_cc_node_ids = self.cc_node_ids[subgraph_a_idx]
        subgraph_b_cc_node_ids = self.cc_node_ids[subgraph_b_idx]

        # Extract fusion-to-CC paths
        subgraph_a_fus_cc_paths = self.extract_subgraph_path_data(self.fus_cc_paths, subgraph_a_idx)
        subgraph_b_fus_cc_paths = self.extract_subgraph_path_data(self.fus_cc_paths, subgraph_b_idx)

        # Extract path influence scores
        subgraph_a_fus_cc_path_pos_influence_scores = self.extract_subgraph_path_data(self.fus_cc_path_pos_influence_scores, subgraph_a_idx)
        subgraph_b_fus_cc_path_pos_influence_scores = self.extract_subgraph_path_data(self.fus_cc_path_pos_influence_scores, subgraph_b_idx)
                
        return (disease_pair,
                disease_pair_label,
                subgraph_a_idx,
                subgraph_a_cc_node_ids,
                subgraph_a_fus_cc_paths,
                subgraph_a_fus_cc_path_pos_influence_scores,
                subgraph_b_idx,
                subgraph_b_cc_node_ids,
                subgraph_b_fus_cc_paths,
                subgraph_b_fus_cc_path_pos_influence_scores)


    def extract_subgraph_path_data(self, fus_cc_path_data, subgraph_idx):
        """
        Extract per-subgraph path tensors.
        """
        
        return {
            path_length: path_tensor[subgraph_idx]
            for path_length, path_tensor in fus_cc_path_data.items()}
        
