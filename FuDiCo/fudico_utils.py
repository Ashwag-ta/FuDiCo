# General Import
import sys

# Sci-kit Learn Import 
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

# PyTorch and PyTorch Geometric Imports
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

# Networkx Import
import networkx as nx



def process_disease_pairs(pos_pairs, neg_pairs, sample_fraction):
    """
    Sample negative disease pairs and combine with positive disease pairs.
    """
  
    # Sample negative pairs
    num_neg_samples = int(neg_pairs.size(0) * sample_fraction)
    sampled_neg_pairs = neg_pairs[torch.randperm(neg_pairs.size(0))[:num_neg_samples]]
  
    # Combine pairs and create labels
    disease_pairs = torch.cat([pos_pairs, sampled_neg_pairs], dim=0).t()
    disease_pair_labels = torch.cat([
        torch.ones(pos_pairs.size(0), dtype=torch.long),  
        torch.zeros(sampled_neg_pairs.size(0), dtype=torch.long)], dim=0)

    perm = torch.randperm(disease_pairs.size(1))  
    disease_pairs = disease_pairs[:, perm]
    disease_pair_labels = disease_pair_labels[perm]
    
    return disease_pairs, disease_pair_labels
    

def read_disease_pairs(subgraphs_file, disease_pair_file, train_ratio=0.8, val_ratio=0.10, test_ratio=0.10):  
    """
    Read disease pairs and split them into train, validation, and test sets.
    """

    # Build disease-pair graph
    disease_pair_graph = nx.Graph()
    with open(subgraphs_file) as sub_f:
        for line in sub_f:
            subgraph_idx = int(line.split("\t")[1].strip())
            disease_pair_graph.add_node(subgraph_idx)
    disease_pair_graph.add_edges_from(nx.read_edgelist(disease_pair_file, nodetype=int).edges())

    # Convert graph to a PyTorch Geometric Data object
    edge_index = torch.tensor(list(disease_pair_graph.edges()), dtype=torch.long).t().contiguous()
    num_nodes = disease_pair_graph.number_of_nodes()
    edge_label = torch.ones(edge_index.size(1), dtype=torch.long)  
    data = Data(edge_index=edge_index, num_nodes=num_nodes, edge_label=edge_label)

    # Split positive and negative pairs
    transform = RandomLinkSplit(is_undirected=True, split_labels=True, num_val=val_ratio, num_test=test_ratio)
    train_data, val_data, test_data = transform(data)

    train_pos_pairs = train_data.pos_edge_label_index.t()
    train_neg_pairs = train_data.neg_edge_label_index.t()
    val_pos_pairs = val_data.pos_edge_label_index.t()
    val_neg_pairs = val_data.neg_edge_label_index.t()
    test_pos_pairs = test_data.pos_edge_label_index.t()
    test_neg_pairs = test_data.neg_edge_label_index.t()

    train_disease_pairs, train_disease_pair_labels = process_disease_pairs(train_pos_pairs, train_neg_pairs, sample_fraction=0.25)
    val_disease_pairs, val_disease_pair_labels = process_disease_pairs(val_pos_pairs, val_neg_pairs, sample_fraction=0.25)
    test_disease_pairs, test_disease_pair_labels = process_disease_pairs(test_pos_pairs, test_neg_pairs, sample_fraction=0.25)

    return (train_disease_pairs, train_disease_pair_labels,
            val_disease_pairs, val_disease_pair_labels,
            test_disease_pairs, test_disease_pair_labels)
   

def assign_subgraphs_to_splits(subgraphs_file):
    """
    Read subgraphs and assign them to train, validation, and test sets.
    """

    all_subgraphs = []
    train_subgraphs = []
    val_subgraphs = []
    test_subgraphs = []

    train_subgraphs_dict = {}
    val_subgraphs_dict = {}
    test_subgraphs_dict = {}

    train_subgraphs_indices = []
    val_subgraphs_indices = []
    test_subgraphs_indices = []

    # Read the subgraph file and assign subgraphs to the appropriate split
    with open(subgraphs_file) as sub_f:
        for idx, line in enumerate(sub_f):
            subgraph_node_ids = line.split("\t")[0].split("-")
            subgraph_node_ids = [int(node) for node in subgraph_node_ids]
            subgraph_idx = int(line.split("\t")[1].strip())
            
            all_subgraphs.append(subgraph_node_ids)

            # Assign to train set
            train_subgraphs.append(subgraph_node_ids)
            train_subgraphs_indices.append(idx)
            train_subgraphs_dict.setdefault(subgraph_idx, []).append(subgraph_node_ids)

            # Assign to validation set
            val_subgraphs.append(subgraph_node_ids)
            val_subgraphs_indices.append(idx)
            val_subgraphs_dict.setdefault(subgraph_idx, []).append(subgraph_node_ids)

            # Assign to test set
            test_subgraphs.append(subgraph_node_ids)
            test_subgraphs_indices.append(idx)
            test_subgraphs_dict.setdefault(subgraph_idx, []).append(subgraph_node_ids)
            
    return (all_subgraphs, train_subgraphs, val_subgraphs, test_subgraphs, 
            train_subgraphs_dict, val_subgraphs_dict, test_subgraphs_dict, 
            train_subgraphs_indices, val_subgraphs_indices, test_subgraphs_indices)


def compute_accuracy(pred_scores, disease_pair_labels):
    """
    Compute comorbidity prediction accuracy for disease pairs.
    """
    
    binary_preds = (pred_scores >= 0.5).float()  # Threshold at 0.5
    
    accuracy = accuracy_score(disease_pair_labels.cpu().numpy(), binary_preds.cpu().numpy())
    
    return torch.tensor([accuracy])


def compute_f1_ap_metrics(pred_scores, disease_pair_labels): 
    """
    Compute F1 score and average precision (AP) of comorbidity prediction for disease pairs.
    """
    
    binary_preds = (pred_scores >= 0.5).cpu().numpy().astype(float) # Threshold at 0.5
    
    labels = disease_pair_labels.cpu().numpy()

    # Compute F1 score
    f1 = f1_score(labels, binary_preds, average='binary', zero_division=0)
    
    # Compute average precision (AP)
    ap = average_precision_score(labels, pred_scores.detach().cpu().numpy()) if labels.sum() > 0 else 0.0
    
    return torch.tensor([f1]), torch.tensor([ap])


