# General Imports
import sys
import numpy as np
import math
from typing import Dict
import multiprocessing
from collections import defaultdict
from pathlib import Path

# Sci-kit Learn Import
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parameter import Parameter

# Pytorch Lightning Import
import lightning.pytorch as pl

# Networkx Import
import networkx as nx

# Ours
sys.path.insert(0, '..') 
import main_config as config
import fudico_utils
from disease_pair_dataset import DiseasePairDataset
from gru_cell import GRUCell



# =====================================================
# Fusion influence-aware GRU path encoder model
# =====================================================

class FusionInfluenceGRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_cell = GRUCell(self.input_size, self.hidden_size)
        
    def forward(self, path_node_embeddings, path_pos_influence_scores, gamma):
        """
        Encode each path by sequentially combining node embeddings and diffusion-based influence scores using a GRU.
        """
        
        num_paths, num_nodes_in_path, _ = path_node_embeddings.shape
        accumulated_influence_score = None
        gamma = gamma.to(path_pos_influence_scores).clamp(1e-6, 1 - 1e-6)
        gamma_complement = 1.0 - gamma
        hidden_state = path_node_embeddings.new_zeros(num_paths, self.hidden_size)

        for path_pos in range(num_nodes_in_path):
            node_embedding = path_node_embeddings[:, path_pos, :] 
            pos_influence_score = path_pos_influence_scores[:, path_pos, :]    

            if path_pos == 0:
                accumulated_influence_score = gamma_complement * pos_influence_score 
            else:
                accumulated_influence_score = gamma * accumulated_influence_score + gamma_complement * pos_influence_score 

            hidden_state = self.gru_cell(node_embedding, hidden_state, pos_influence_score, accumulated_influence_score)

        return hidden_state, accumulated_influence_score     



# =====================================================
# FuDiCo model
# =====================================================

class FuDiCo(pl.LightningModule):
    def __init__(self, hyperparameters: Dict, PPI_network_path: str, disease_pairs_path: str, subgraphs_path: str,
                 node_embeddings_path: str, fusion_pairs_path: str, fusion_diffusion_data_path: str):
        super().__init__()
        
        # Model hyperparameters
        self.hyperparameters = hyperparameters
        
        # Input paths
        self.PPI_network_path = PPI_network_path
        self.disease_pairs_path = disease_pairs_path
        self.subgraphs_path = subgraphs_path
        self.node_embeddings_path = node_embeddings_path
        self.fusion_pairs_path = fusion_pairs_path
        self.fusion_diffusion_data_path = Path(fusion_diffusion_data_path)
        
        # Read in data
        self.read_data()

        # Core configs
        self.input_size = self.hyperparameters['init_node_dim']
        self.max_paths_per_cc = self.hyperparameters['max_paths_per_cc']
        self.max_path_length = self.hyperparameters['max_path_length']
        self.used_path_lengths = self.hyperparameters['used_path_lengths']
        self.num_path_lengths = len(self.used_path_lengths)

        # Path-length-specific node projections
        self.node_projections = nn.ModuleList([
            nn.Linear(self.input_size, self.hyperparameters['node_proj_dim'])
            for _ in range(self.num_path_lengths)])

        # Normalize concatenated subgraph embeddings across all path lengths before prediction
        self.final_subgraph_embedding_norm = nn.LayerNorm(self.hyperparameters['feature_dim'] * self.num_path_lengths)

        # Projects CC embeddings to the final feature space
        self.cc_projection = nn.Sequential(
            nn.Linear(self.hyperparameters['node_proj_dim'], self.hyperparameters['cc_hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.hyperparameters['cc_hidden_dim'], self.hyperparameters['feature_dim']))

        # Fusion influence-aware GRU path encoder
        self.path_encoder = FusionInfluenceGRUEncoder(self.hyperparameters['node_proj_dim'], self.hyperparameters['path_hidden_dim'])

        # Learnable propagation parameter for accumulated influence along each path
        self.raw_gamma = nn.Parameter(torch.tensor(self.hyperparameters['gamma']))  

        # BCE loss for comorbid disease pair prediction
        self.loss = nn.BCELoss() 

        # Validation metric storage
        self.val_metric_scores = []

        # Learnable no-path gate vector per path length
        self.no_path_gate_vectors = nn.ParameterList([
            nn.Parameter(torch.full((self.hyperparameters['feature_dim'],), float(self.hyperparameters['no_path_gate_init'])))   
            for _ in range(self.num_path_lengths)])

        # MLP predictor that takes concatenated subgraph embeddings and outputs comorbidity probability
        self.predictor = nn.Sequential(
            nn.Linear(self.hyperparameters['feature_dim'] * self.num_path_lengths * 2, self.hyperparameters['feature_dim']),
            nn.ReLU(),
            nn.Linear(self.hyperparameters['feature_dim'], 1),
            nn.Sigmoid())
      

    def encode_paths_for_path_length(self, fus_cc_paths, fus_cc_path_pos_influence_scores, node_embeddings_by_path_length, path_length, path_aggregator, device):
        """
        Encode fusion-to-CC paths for one path length and aggregate them into CC embeddings.
        """
        
        cc_path_node_ids = fus_cc_paths[path_length].squeeze(0).to(device)
        cc_path_pos_influence_scores = fus_cc_path_pos_influence_scores[path_length].to(device)
        num_subgraphs, num_ccs, num_paths, num_nodes_in_path = cc_path_node_ids.shape

        # Select valid paths
        valid_path_mask = (cc_path_node_ids != config.PAD_VALUE).any(dim=-1).view(-1) 
        valid_path_idx  = valid_path_mask.nonzero(as_tuple=False).squeeze(-1).to(device, dtype=torch.long)
        valid_paths = cc_path_node_ids.view(-1, num_nodes_in_path).index_select(0, valid_path_idx)            
        
        # Retrieve node embeddings and influence scores for valid paths
        valid_path_node_ids_flat = valid_paths.reshape(-1).to(device, dtype=torch.long)
        valid_path_embeddings = node_embeddings_by_path_length.index_select(0, valid_path_node_ids_flat).view(-1, num_nodes_in_path, node_embeddings_by_path_length.size(1))
        valid_path_pos_influence_scores = cc_path_pos_influence_scores.view(-1, num_nodes_in_path).index_select(0, valid_path_idx).unsqueeze(-1)
        valid_path_to_cc_index = (valid_path_idx // num_paths).to(device, dtype=torch.long)
     
        # Encode valid paths with GRU
        encoded_valid_paths, valid_path_influence_scores = self.path_encoder(valid_path_embeddings, valid_path_pos_influence_scores, torch.sigmoid(self.raw_gamma))

        # Aggregate encoded paths into CC embeddings
        cc_path_embedding_sum = torch.zeros(num_subgraphs * num_ccs, encoded_valid_paths.size(1), device=device, dtype=encoded_valid_paths.dtype)
        cc_path_count = torch.zeros(num_subgraphs * num_ccs, 1, device=device, dtype=valid_path_influence_scores.dtype)
        cc_path_count.index_add_(0, valid_path_to_cc_index, torch.ones_like(valid_path_influence_scores, device=device))

        if path_aggregator == 'mean':
            cc_path_embedding_sum.index_add_(0, valid_path_to_cc_index, encoded_valid_paths)
            cc_path_agg_embeddings = (cc_path_embedding_sum  / cc_path_count.clamp_min(1.0)).view(num_subgraphs, num_ccs, encoded_valid_paths.size(1))

        elif path_aggregator == 'sum':
            cc_path_embedding_sum.index_add_(0, valid_path_to_cc_index, encoded_valid_paths)
            cc_path_agg_embeddings = cc_path_embedding_sum.view(num_subgraphs, num_ccs, encoded_valid_paths.size(1))

        elif path_aggregator == 'attn_softmax':
            path_influence_scores = torch.full((num_subgraphs, num_ccs, num_paths), float('-inf'), device=device, dtype=valid_path_influence_scores.dtype)
            path_influence_scores.view(-1).index_put_((valid_path_idx,), valid_path_influence_scores.squeeze(-1))
            path_attn_weights = torch.softmax(path_influence_scores, dim=-1)                                               
            valid_path_attn_weights = (path_attn_weights.view(-1).index_select(0, valid_path_idx).unsqueeze(-1)) 
            cc_path_embedding_sum.index_add_(0, valid_path_to_cc_index, encoded_valid_paths * valid_path_attn_weights)
            cc_path_agg_embeddings = cc_path_embedding_sum.view(num_subgraphs, num_ccs, encoded_valid_paths.size(1))

        # Boolean mask indicating which CCs have at least one valid path
        cc_has_path_mask = (cc_path_count.view(num_subgraphs, num_ccs, 1) > 0)
        
        return cc_has_path_mask, cc_path_agg_embeddings


    def forward(self, cc_node_ids, fus_cc_paths, fus_cc_path_pos_influence_scores):
        """
        Encode each subgraph from its connected components and fusion-to-CC paths.
        """

        device = self.node_embeddings.weight.device
        cc_node_ids = cc_node_ids.to(device)
        subgraph_embeddings_all_path_lengths = []

        # Masks for valid CCs and valid nodes inside each CC
        cc_mask = (cc_node_ids != config.PAD_VALUE)[:, :, 0].to(device) 
        cc_node_mask = (cc_node_ids != config.PAD_VALUE).unsqueeze(-1) 

        for path_length_idx, path_length in enumerate(self.used_path_lengths):
            # Project node embeddings for this path length
            node_embeddings_by_path_length = self.node_projections[path_length_idx](self.node_embeddings.weight)

            # Retrieve node embeddings for each CC 
            cc_node_embeddings = node_embeddings_by_path_length[cc_node_ids]  
            cc_node_embeddings = cc_node_embeddings * cc_node_mask.to(cc_node_embeddings.dtype)
            
            # Aggregate node embeddings into initial CC embeddings
            if self.hyperparameters['cc_aggregator'] == 'sum':
                init_cc_embeddings = torch.sum(cc_node_embeddings, dim=2)     
                
            elif self.hyperparameters['cc_aggregator'] == 'max':
                init_cc_embeddings = torch.max(cc_node_embeddings, dim=2)[0]  
                
            elif self.hyperparameters['cc_aggregator'] == 'mean':
                cc_node_counts = cc_node_mask.squeeze(-1).sum(dim=2, keepdim=True).clamp(min=1)  
                init_cc_embeddings = cc_node_embeddings.sum(dim=2) / cc_node_counts 

            # Encode and aggregate fusion-to-CC paths
            cc_has_path_mask, cc_path_agg_embeddings = self.encode_paths_for_path_length(fus_cc_paths,
                                                                                         fus_cc_path_pos_influence_scores,
                                                                                         node_embeddings_by_path_length,
                                                                                         path_length,
                                                                                         self.hyperparameters['path_aggregator'],
                                                                                         device)

            # Project CC embeddings (initial and path-based) to the shared feature space
            init_cc_projected_embeddings = self.cc_projection(init_cc_embeddings)  
            cc_path_projected_embeddings = self.cc_projection(cc_path_agg_embeddings)  

            # Learnable gate for CCs without paths (per feature dimension)
            no_path_gate = torch.sigmoid(self.no_path_gate_vectors[path_length_idx]).to(device=cc_path_projected_embeddings.device, dtype=cc_path_projected_embeddings.dtype).view(1, 1, -1)  

            # Use path-based CC embeddings when paths exist; otherwise use gated initial CC embeddings
            cc_embeddings_by_path_length = torch.where(cc_has_path_mask.bool(), cc_path_projected_embeddings, no_path_gate * init_cc_projected_embeddings)
            cc_embeddings_by_path_length = cc_embeddings_by_path_length * cc_mask.unsqueeze(-1) 

            # Aggregate CC embeddings into one subgraph embedding for this path length
            subgraph_embeddings_for_path_length = (cc_embeddings_by_path_length.sum(dim=1) / cc_mask.sum(dim=1, keepdim=True).to(cc_embeddings_by_path_length.dtype).clamp_min(1))

            subgraph_embeddings_all_path_lengths.append(subgraph_embeddings_for_path_length) 

        # Concatenate embeddings from all path lengths to form final subgraph embeddings
        subgraph_embeddings = torch.cat(subgraph_embeddings_all_path_lengths, dim=1)    
        
        # Normalize before disease-pair prediction
        subgraph_embeddings = self.final_subgraph_embedding_norm(subgraph_embeddings)
    
        return subgraph_embeddings   
               

                
# =====================================================
# Training, validation, and testing
# =====================================================

    def on_train_epoch_start(self):
        """
        Reset storage for training outputs at the start of the epoch.
        """
        
        print("\n\n...Start training epoch...", flush=True)
        
        self.train_step_outputs = []  

        
    def training_step(self, train_batch, batch_idx): 
        """
        Run one training step on a batch.
        """
        
        # Extract batch inputs
        disease_pairs = train_batch['disease_pairs']
        disease_pair_labels = train_batch['disease_pair_labels'].squeeze(-1)
        subgraph_idx = train_batch['subgraph_idx']
        cc_node_ids = train_batch['cc_node_ids']
        fus_cc_paths = train_batch['fus_cc_paths']
        fus_cc_path_pos_influence_scores = train_batch['fus_cc_path_pos_influence_scores']
        
        # Forward pass
        subgraph_embeddings = self.forward(cc_node_ids, fus_cc_paths, fus_cc_path_pos_influence_scores)
                                           
        # Get embeddings for each disease subgraph pair
        pair_pos_in_subgraph = torch.searchsorted(subgraph_idx, disease_pairs) # Map pair IDs to subgraph embedding positions
        subgraph_a_emb = subgraph_embeddings[pair_pos_in_subgraph[:, 0]]
        subgraph_b_emb = subgraph_embeddings[pair_pos_in_subgraph[:, 1]]

        # Concatenate and pass through MLP
        pair_emb = torch.cat([subgraph_a_emb, subgraph_b_emb], dim=1)
        pred_scores = self.predictor(pair_emb).squeeze()  

        # Compute loss and metrics
        loss = self.loss(pred_scores, disease_pair_labels.float())  
        accuracy = fudico_utils.compute_accuracy(pred_scores, disease_pair_labels)
        f1, average_precision = fudico_utils.compute_f1_ap_metrics(pred_scores, disease_pair_labels)

        train_step_results = {'train_loss': loss,
                              'train_accuracy': accuracy,
                              'train_f1': f1,
                              'train_average_precision': average_precision,
                              'train_pred_scores': pred_scores,
                              'train_labels': disease_pair_labels}
        
        # Log metrics
        self.log_dict({'train_loss': loss,
                       'train_accuracy': accuracy,
                       'train_f1': f1,
                       'train_average_precision': average_precision}, batch_size=self.hyperparameters['batch_size'], on_step=False, on_epoch=True, prog_bar=False)
            
        self.train_step_outputs.append(train_step_results)

        return {'loss': loss,
                'train_accuracy': accuracy,
                'train_f1': f1,
                'train_average_precision': average_precision,
                'train_pred_scores': pred_scores,
                'train_labels': disease_pair_labels}
        
     
    def on_train_epoch_end(self):
        """
        Aggregate metrics at the end of the training epoch.
        """

        avg_loss = self.trainer.callback_metrics['train_loss']
        avg_accuracy = self.trainer.callback_metrics['train_accuracy']
        avg_f1 = self.trainer.callback_metrics['train_f1']
        avg_average_precision = self.trainer.callback_metrics['train_average_precision']

        print("\n\n...Train epoch results...", flush=True)
        print(f"Train epoch loss: {avg_loss:.4f}")
        print(f"Train epoch accuracy: {avg_accuracy:.4f}")
        print(f"Train epoch F1: {avg_f1:.4f}")
        print(f"Train epoch average precision (AP): {avg_average_precision:.4f}")

        # Compute AUROC for the entire training epoch
        pred_scores = torch.cat([x['train_pred_scores'] for x in self.train_step_outputs], dim=0)
        labels = torch.cat([x['train_labels'] for x in self.train_step_outputs], dim=0)
        auroc = roc_auc_score(labels.detach().cpu().numpy(), pred_scores.detach().cpu().numpy())
        
        print(f"Train epoch AUROC: {auroc:.4f}")
        self.log_dict({'train_auroc': auroc}, prog_bar=False)
        
        self.train_step_outputs.clear()

        print("...End training epoch...\n")
        
        return {'train_auroc': auroc}


    def on_validation_epoch_start(self):   
        """
        Reset storage for validation outputs at the start of the epoch.
        """
        
        print("\n\n...Start validation epoch...", flush=True)
        
        self.val_step_outputs = [] 


    def validation_step(self, val_batch, batch_idx):
        """
        Run one validation step on a batch.
        """
        
        # Extract batch inputs
        disease_pairs = val_batch['disease_pairs']
        disease_pair_labels = val_batch['disease_pair_labels'].squeeze(-1)
        subgraph_idx = val_batch['subgraph_idx']
        cc_node_ids = val_batch['cc_node_ids']
        fus_cc_paths = val_batch['fus_cc_paths']
        fus_cc_path_pos_influence_scores = val_batch['fus_cc_path_pos_influence_scores']

        # Forward pass
        subgraph_embeddings = self.forward(cc_node_ids, fus_cc_paths, fus_cc_path_pos_influence_scores)

        # Get embeddings for each disease pair
        pair_pos_in_subgraph = torch.searchsorted(subgraph_idx, disease_pairs)
        subgraph_a_emb = subgraph_embeddings[pair_pos_in_subgraph[:, 0]]
        subgraph_b_emb = subgraph_embeddings[pair_pos_in_subgraph[:, 1]]

        # Concatenate and pass through MLP
        pair_emb = torch.cat([subgraph_a_emb, subgraph_b_emb], dim=1)
        pred_scores = self.predictor(pair_emb).squeeze()

        # Compute loss and metrics
        loss = self.loss(pred_scores, disease_pair_labels.float()) 
        accuracy = fudico_utils.compute_accuracy(pred_scores, disease_pair_labels)
        f1, average_precision = fudico_utils.compute_f1_ap_metrics(pred_scores, disease_pair_labels)
        
        val_step_results = {'val_pred_scores': pred_scores,
                            'val_labels': disease_pair_labels}
        
        # Log metrics
        self.log_dict({'val_loss': loss, 
                       'val_accuracy': accuracy,
                       'val_f1': f1,
                       'val_average_precision': average_precision}, batch_size=self.hyperparameters['batch_size'], on_step=False, on_epoch=True, prog_bar=False)
        
        self.val_step_outputs.append(val_step_results)
        
        return val_step_results

       
    def on_test_epoch_start(self):
        """
        Reset storage for testing outputs at the start of the epoch.
        """

        print("\n\n...Start testing epoch...", flush=True)
        
        self.test_step_outputs = []  

    
    def test_step(self, test_batch, batch_idx):
        """
        Run one testing step on a batch.
        """
                
        # Extract batch inputs
        disease_pairs = test_batch['disease_pairs']
        disease_pair_labels = test_batch['disease_pair_labels'].squeeze(-1)
        subgraph_idx = test_batch['subgraph_idx']
        cc_node_ids = test_batch['cc_node_ids']
        fus_cc_paths = test_batch['fus_cc_paths']
        fus_cc_path_pos_influence_scores = test_batch['fus_cc_path_pos_influence_scores']

        # Forward pass
        subgraph_embeddings = self.forward(cc_node_ids, fus_cc_paths, fus_cc_path_pos_influence_scores)

        # Get embeddings for each disease pair
        pair_pos_in_subgraph = torch.searchsorted(subgraph_idx, disease_pairs)  # Map pair IDs to subgraph embedding positions
        subgraph_a_emb = subgraph_embeddings[pair_pos_in_subgraph[:, 0]]
        subgraph_b_emb = subgraph_embeddings[pair_pos_in_subgraph[:, 1]]

        # Concatenate and pass through MLP
        pair_emb = torch.cat([subgraph_a_emb, subgraph_b_emb], dim=1)  
        pred_scores = self.predictor(pair_emb).squeeze()  
        
        # Compute loss and metrics
        loss = self.loss(pred_scores, disease_pair_labels.float()) 
        accuracy = fudico_utils.compute_accuracy(pred_scores, disease_pair_labels)
        f1, average_precision = fudico_utils.compute_f1_ap_metrics(pred_scores, disease_pair_labels)
        
        test_step_results = {'test_pred_scores': pred_scores,
                             'test_labels': disease_pair_labels}

        # Log metrics
        self.log_dict({'test_loss': loss, 
                       'test_accuracy': accuracy,
                       'test_f1': f1,
                       'test_average_precision': average_precision}, batch_size=self.hyperparameters['batch_size'], on_step=False, on_epoch=True, prog_bar=False)

        self.test_step_outputs.append(test_step_results)

        return test_step_results


    def on_validation_epoch_end(self):
        """
        Aggregate metrics at the end of the validation epoch.
        """
        
        avg_loss = self.trainer.callback_metrics['val_loss']
        avg_accuracy = self.trainer.callback_metrics['val_accuracy']
        avg_f1 = self.trainer.callback_metrics['val_f1']
        avg_average_precision = self.trainer.callback_metrics['val_average_precision']

        print("\n\n...Validation epoch results...", flush=True)
        print(f"Validation epoch loss: {avg_loss:.4f}")
        print(f"Validation epoch accuracy: {avg_accuracy:.4f}")
        print(f"Validation epoch F1: {avg_f1:.4f}")
        print(f"Validation epoch average precision: {avg_average_precision:.4f}")

        # Compute AUROC for the entire validation epoch
        pred_scores = torch.cat([x['val_pred_scores'] for x in self.val_step_outputs], dim=0)
        labels = torch.cat([x['val_labels'] for x in self.val_step_outputs], dim=0)
        auroc = roc_auc_score(labels.detach().cpu().numpy(), pred_scores.detach().cpu().numpy())

        print(f"Validation epoch AUROC: {auroc:.4f}")
        self.log_dict({'val_auroc': auroc}, prog_bar=False)
        
        val_epoch_metrics = {'epoch': self.current_epoch, 
                             'val_loss': avg_loss.item(),
                             'val_accuracy': avg_accuracy.item(),
                             'val_f1': avg_f1.item(),
                             'val_average_precision': avg_average_precision.item(),
                             'val_auroc': auroc}
        
        self.val_metric_scores.append(val_epoch_metrics)
        
        self.val_step_outputs.clear()
        
        print("...End validation epoch...\n")


    def on_test_epoch_end(self):
        """
        Aggregate metrics at the end of the testing epoch.
        """
        
        avg_loss = self.trainer.callback_metrics['test_loss']
        avg_accuracy = self.trainer.callback_metrics['test_accuracy']
        avg_f1 = self.trainer.callback_metrics['test_f1']
        avg_average_precision = self.trainer.callback_metrics['test_average_precision']

        # Compute AUROC for the entire testing epoch
        pred_scores = torch.cat([x['test_pred_scores'] for x in self.test_step_outputs], dim=0)
        labels = torch.cat([x['test_labels'] for x in self.test_step_outputs], dim=0)
        auroc = roc_auc_score(labels.detach().cpu().numpy(), pred_scores.detach().cpu().numpy())
        
        self.log_dict({'test_auroc_epoch': auroc}, prog_bar=False)

        # ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(labels.detach().cpu().numpy(), pred_scores.detach().cpu().numpy())
       
        # Precision-Recall Curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels.detach().cpu().numpy(), pred_scores.detach().cpu().numpy())

        self.test_epoch_metrics = {'epoch': self.current_epoch, 
                                   'test_loss': avg_loss.item(),
                                   'test_accuracy': avg_accuracy.item(),
                                   'test_f1': avg_f1.item(),
                                   'test_average_precision': avg_average_precision.item(),
                                   'test_auroc': auroc,
                                   'fpr': fpr.tolist(),
                                   'tpr': tpr.tolist(),
                                   'roc_thresholds': roc_thresholds.tolist(),
                                   'precision_curve': precision_curve.tolist(),
                                   'recall_curve': recall_curve.tolist(),
                                   'pr_thresholds': pr_thresholds.tolist(),
                                   'pred_scores': pred_scores,
                                   'labels': labels}

        self.test_step_outputs.clear()

        return self.test_epoch_metrics



# =====================================================
# Read data 
# =====================================================

    def reindex_data(self, data, mode: str):
        """
        Reindex node IDs to be 1-indexed.

        Args:
            data: Input (list or dict) representing subgraphs, fusion gene pairs, or a dictionary of subgraphs.
            mode: One of ['subgraph', 'pair', 'dict'].

        Returns:
            Reindexed data.
        """

        if mode == 'subgraph':
            # List of subgraphs, where each subgraph is a list of node IDs
            return [[node_id + 1 for node_id in subgraph_node_ids] for subgraph_node_ids in data]

        elif mode == 'pair':
            # List of fusion gene pairs (node ID pairs)
            return [[node_a + 1, node_b + 1] for node_a, node_b in data]

        elif mode == 'dict':
            # Dictionary where values are lists of subgraphs (each subgraph is a list of node IDs)
            return {
                subgraph_id: [[node_id + 1 for node_id in subgraph_node_ids] for subgraph_node_ids in subgraph_list]
                for subgraph_id, subgraph_list in data.items()
                   }
        

    def read_data(self):
        """
        Read the input PPI network, disease pairs, disease subgraphs, fusion gene pairs, and pretrained node embeddings.
        """

        print("...Reading data...")

        # Read the PPI network
        self.PPI_graph = nx.read_edgelist(config.PROJECT_ROOT / self.PPI_network_path)

        # Read and split the disease pairs into train, validation, and test sets
        self.train_disease_pairs, self.train_disease_pair_labels, self.val_disease_pairs, self.val_disease_pair_labels, self.test_disease_pairs, self.test_disease_pair_labels =\
                                                        fudico_utils.read_disease_pairs(config.PROJECT_ROOT / self.subgraphs_path, config.PROJECT_ROOT / self.disease_pairs_path)
                                                                                                    
        self.all_subgraphs, self.train_subgraphs, self.val_subgraphs, self.test_subgraphs, self.train_subgraphs_dict, self.val_subgraphs_dict, self.test_subgraphs_dict,\
                                                        self.train_subgraphs_indices, self.val_subgraphs_indices, self.test_subgraphs_indices = fudico_utils.assign_subgraphs_to_splits(
                                                        config.PROJECT_ROOT / self.subgraphs_path)

        # Read fusion gene pairs
        fus_pairs = []
        with open(config.PROJECT_ROOT / self.fusion_pairs_path, 'r') as fus_file:
            for fus_pair in fus_file:
                fus_gene_a, fus_gene_b = map(int, fus_pair.strip().split())
                fus_pairs.append([fus_gene_a, fus_gene_b])

        # Reindex fusion gene pairs to be 1-indexed
        fus_pairs = self.reindex_data(fus_pairs, mode="pair")
        self.fus_pairs = np.array(fus_pairs, dtype=np.int32)
        
        # Reindex PPI graph nodes to be 1-indexed
        PPI_node_mapping  = {n:int(n)+1 for n in self.PPI_graph.nodes()}
        self.PPI_graph = nx.relabel_nodes(self.PPI_graph, PPI_node_mapping )
        
        # Reindex subgraphs to be 1-indexed
        self.train_subgraphs = self.reindex_data(self.train_subgraphs, mode="subgraph")
        self.val_subgraphs = self.reindex_data(self.val_subgraphs, mode="subgraph")
        self.test_subgraphs = self.reindex_data(self.test_subgraphs, mode="subgraph")
        self.all_subgraphs = self.reindex_data(self.all_subgraphs, mode="subgraph")

        # Reindex subgraph dictionaries to be 1-indexed
        self.train_subgraphs_dict = self.reindex_data(self.train_subgraphs_dict, mode="dict")
        self.val_subgraphs_dict = self.reindex_data(self.val_subgraphs_dict, mode="dict")
        self.test_subgraphs_dict = self.reindex_data(self.test_subgraphs_dict, mode="dict")

        # Load pretrained ESM-2 node embeddings
        pretrained_node_embeddings = torch.load(config.PROJECT_ROOT / self.node_embeddings_path, map_location=self.device)
        pad_embedding = torch.zeros(1, pretrained_node_embeddings.shape[1], device=pretrained_node_embeddings.device)
        node_embeddings_with_pad = torch.cat((pad_embedding, pretrained_node_embeddings), dim=0)
        self.node_embeddings = nn.Embedding.from_pretrained(node_embeddings_with_pad, freeze=self.hyperparameters['freeze_node_embeds'], padding_idx=config.PAD_VALUE)

        print("...Finished reading data...")



# =====================================================
# Initialize subgraph connected components
# =====================================================

    def initialize_subgraph_ccs(self, subgraph_node_ids):
        """
        Initialize connected components for each subgraph as a padded tensor of node IDs.

        Args:
            - subgraph_node_ids (list): List of subgraphs, where each subgraph is a list of node IDs.

        Returns:
            - Tensor: Padded tensor of shape [num_subgraphs, max_num_cc, max_num_nodes_per_cc].
        """

        num_subgraphs = len(subgraph_node_ids) 
        cc_node_id_list = []
        
        for curr_subgraph_node_ids in subgraph_node_ids:
            subgraph = nx.subgraph(self.PPI_graph, curr_subgraph_node_ids) # Extract subgraph from the PPI graph
            connected_components = list(nx.connected_components(subgraph)) # Connected components in the subgraph
            cc_node_id_list.append([torch.LongTensor(list(cc_ids)) for cc_ids in connected_components])

        # Pad the number of connected components across subgraphs
        max_num_cc = max([len(cc_list) for cc_list in cc_node_id_list]) 
        for cc_list in cc_node_id_list:
            while len(cc_list) < max_num_cc:
                cc_list.append(torch.LongTensor([config.PAD_VALUE]))

        # Pad the number of nodes within connected components
        all_padded_cc_node_ids = [cc_node_ids for cc_list in cc_node_id_list for cc_node_ids in cc_list]
        assert len(all_padded_cc_node_ids) % max_num_cc == 0
        cc_node_ids_padded = pad_sequence(all_padded_cc_node_ids, batch_first=True, padding_value=config.PAD_VALUE)
        cc_node_ids_padded = cc_node_ids_padded.view(num_subgraphs, max_num_cc, -1)

        return cc_node_ids_padded 


   
# =====================================================
# Get fusion gene sets for each subgraph
# =====================================================
 
    def initialize_subgraph_fus_sets(self, file_name, cc_node_ids, fus_pairs):
        """
        Compute internal and border fusion gene sets for each subgraph.

        Args:
            file_name (Path): Output file path.
            cc_node_ids (Tensor): Tensor of shape [num_subgraphs, max_num_cc, max_num_nodes_per_cc].
            fus_pairs (List[Tuple[int, int]]): List of fusion gene pairs.

        Returns:
            Dict: Mapping each subgraph index to a dictionary with:
                - "fusion_genes": unique fusion genes associated with the subgraph
                - "internal": fusion gene pairs fully inside the subgraph
                - "border": fusion gene pairs with exactly one gene inside the subgraph
        """
        
        subgraph_fus_sets = defaultdict(dict)
        
        for s_idx, subgraph in enumerate(cc_node_ids):
            subgraph_node_ids = subgraph.reshape(-1)
            non_padded_subgraph_node_ids = np.unique(subgraph_node_ids[subgraph_node_ids != config.PAD_VALUE].cpu().numpy())
            fus_gene_a_in_subgraph = np.isin(fus_pairs[:, 0], non_padded_subgraph_node_ids)
            fus_gene_b_in_subgraph = np.isin(fus_pairs[:, 1], non_padded_subgraph_node_ids)
            
            # Extract subgraph internal and border fusion genes
            subgraph_internal_fus = fus_pairs[fus_gene_a_in_subgraph & fus_gene_b_in_subgraph]
            subgraph_border_fus = fus_pairs[fus_gene_a_in_subgraph ^ fus_gene_b_in_subgraph]
           
            subgraph_fus_genes = np.unique(np.concatenate([np.array(subgraph_internal_fus).ravel(), np.array(subgraph_border_fus).ravel()]))

            subgraph_fus_sets[s_idx] = {"fusion_genes": subgraph_fus_genes.tolist(), "internal": subgraph_internal_fus.tolist(), "border": subgraph_border_fus.tolist()}

        np.save(file_name, np.array(subgraph_fus_sets, dtype=object), allow_pickle=True)
        
        return subgraph_fus_sets


    def get_subgraph_fus_sets(self, split):
        """ 
        Compute fusion gene sets for subgraphs in the specified split.
        """

        train_subgraph_fus_set_path = self.fusion_diffusion_data_path / "train_subgraph_fus_sets.npy"

        val_subgraph_fus_set_path = self.fusion_diffusion_data_path / "val_subgraph_fus_sets.npy"

        test_subgraph_fus_set_path = self.fusion_diffusion_data_path / "test_subgraph_fus_sets.npy"

        if split == 'train_val':
            if train_subgraph_fus_set_path.exists():
                self.train_subgraph_fus_set = np.load(train_subgraph_fus_set_path, allow_pickle=True).item()
            else:
                self.train_subgraph_fus_set = self.initialize_subgraph_fus_sets(train_subgraph_fus_set_path, self.train_cc_node_ids, self.fus_pairs)
            
            if val_subgraph_fus_set_path.exists():
                self.val_subgraph_fus_set = np.load(val_subgraph_fus_set_path, allow_pickle=True).item()
            else:
                self.val_subgraph_fus_set = self.initialize_subgraph_fus_sets(val_subgraph_fus_set_path, self.val_cc_node_ids, self.fus_pairs)

        elif split == 'test':
            if test_subgraph_fus_set_path.exists():
                self.test_subgraph_fus_set = np.load(test_subgraph_fus_set_path, allow_pickle=True).item()
            else:
                self.test_subgraph_fus_set = self.initialize_subgraph_fus_sets(test_subgraph_fus_set_path, self.test_cc_node_ids, self.fus_pairs)



# =====================================================
# Compute diffusion reachability matrices
# =====================================================

    def get_graph_diffusion_reachability(self):
        """
        Compute weighted diffusion reachability matrices.

        Returns:
            dict: mapping each path length k to a weighted diffusion reachability matrix 
                  of shape (num_ppi_nodes, num_ppi_nodes).
        """
        
        beta = float(self.hyperparameters.get('beta'))

        # Normalized path length weights (beta^k)
        path_length_weights = np.array([beta**k for k in range(1, self.max_path_length + 1)], dtype=np.float32)
        path_length_weights /= path_length_weights.sum()
        
        # Load weighted diffusion reachability matrices if all exist
        reachability_matrix_paths = {
            k: self.fusion_diffusion_data_path / f"weighted_diff_reach_up_to_len_{k}_beta{beta:.2f}.pt"
            for k in range(1, self.max_path_length + 1)}
        
        if all(path.exists() for path in reachability_matrix_paths.values()):
            self.weighted_diff_reach = {
                k: torch.load(path)
                for k, path in reachability_matrix_paths.items()}
            return self.weighted_diff_reach

        # Initialize weighted diffusion reachability 
        self.weighted_diff_reach = {}
        weighted_diff_reach_up_to_k = None  # Weighted diffusion reachability up to current k

        for k in range(1, self.max_path_length + 1):
            # Load k-th power of diffusion operator
            diffusion_operator_k = torch.from_numpy(np.load(self.fusion_diffusion_data_path / f"diffusion_operator_k_{k}.npy", mmap_mode="r").astype(np.float32, copy=False))

            # Number of PPI graph nodes (matrix dimension)
            num_nodes = diffusion_operator_k.size(0)

            if weighted_diff_reach_up_to_k is None:
                weighted_diff_reach_up_to_k = torch.zeros(num_nodes + 1, num_nodes + 1, dtype=torch.float32)

            diffusion_operator_k_pad = torch.zeros(num_nodes + 1, num_nodes + 1, dtype=torch.float32) # Convert to 1-based indexing
            diffusion_operator_k_pad[1:, 1:] = diffusion_operator_k
            
            # Update weighted diffusion reachability (up to current k)
            weighted_diff_reach_up_to_k += float(path_length_weights[k - 1]) * diffusion_operator_k_pad

            self.weighted_diff_reach[k] = weighted_diff_reach_up_to_k.clone()
            torch.save(self.weighted_diff_reach[k], reachability_matrix_paths[k])

        return self.weighted_diff_reach



# =====================================================
# Sample fusion-to-CC paths
# =====================================================
  
    def sample_fus_cc_paths_by_k(self, file_name, cc_node_ids, fus_sets, fusion_diffusion_data_path, max_paths_per_cc, used_path_lengths):
        """
        Sample fusion-to-CC paths for each path length k.

        Returns:
            Dict: Mapping each path length k to a tensor of shape [num_subgraphs, num_cc, max_paths_per_cc, num_nodes_in_path].
        """
       
        num_subgraphs, max_num_cc, _ = cc_node_ids.shape

        sampled_fus_cc_paths_by_k = {
            k: torch.full((num_subgraphs, max_num_cc, max_paths_per_cc, k + 1), config.PAD_VALUE, dtype=torch.long)
            for k in used_path_lengths}
                
        for k in used_path_lengths:
    
            print(f"...Loading fusion-to-subgraph paths of length {k}...")
            fus_subgraph_paths_path = fusion_diffusion_data_path / f"fusion_to_subgraph_paths_len_{k}.npy"
            fus_subgraph_paths = np.load(fus_subgraph_paths_path, allow_pickle=True).item()
            
            print(f"...Sampling fusion-to-CC paths of length {k}...")
            for s_idx, subgraph in enumerate(cc_node_ids):
                # Fusion genes associated with the current subgraph
                subgraph_fus_ids = torch.as_tensor(fus_sets[s_idx]["fusion_genes"], dtype=torch.long)
                for cc_idx, component in enumerate(subgraph):
                    cc_node_ids = component[component != config.PAD_VALUE]
                    if cc_node_ids.numel() == 0:
                        continue

                    fus_cc_reach_scores = self.weighted_diff_reach[k].index_select(0, subgraph_fus_ids).index_select(1, cc_node_ids)
                    sorted_fus_cc_reach_scores, sorted_fus_idx = torch.sort(fus_cc_reach_scores, dim=0, descending=True)
                    sorted_fus_ids = subgraph_fus_ids[sorted_fus_idx] 

                    num_fus_nodes, num_cc_nodes = sorted_fus_cc_reach_scores.shape
                    fus_cc_candidates = []
                    for node_idx in range(num_cc_nodes):
                        cc_node_id = int(cc_node_ids[node_idx].item()) - 1  
                        fus_node_candidates = []
                        
                        for fus_idx in range(num_fus_nodes):
                            fus_node_reach_score = float(sorted_fus_cc_reach_scores[fus_idx, node_idx].item())
                            if fus_node_reach_score <= 0.0:
                                break  # remaining rows for this node are <= 0
                            
                            fus_id = int(sorted_fus_ids[fus_idx, node_idx].item()) - 1  
                            fus_node_candidate_paths = fus_subgraph_paths.get((fus_id, cc_node_id), [])
                            
                            if fus_node_candidate_paths:
                                fus_node_candidates.append((fus_node_reach_score, fus_id, cc_node_id, fus_node_candidate_paths))  

                        if fus_node_candidates:
                            fus_cc_candidates.append(fus_node_candidates)
      
                    # Coverage sampling step: one best path per component node
                    num_selected_paths = 0
                    used_fus_node_pairs = set()  

                    for fus_node_candidates in fus_cc_candidates:  
                        if num_selected_paths >= max_paths_per_cc:
                            break

                        fus_node_reach_score, fus_id, cc_node_id, fus_node_candidate_paths = fus_node_candidates[0]   # best candidate for this node
    
                        path_pos_influence_scores = self.compute_path_pos_influence_scores((torch.tensor(fus_node_candidate_paths, dtype=torch.long).add_(1).unsqueeze(0).unsqueeze(0)), k)
                        path_influence_scores = path_pos_influence_scores.mean(dim=-1).view(-1)
                        best_path_idx = int(path_influence_scores.argmax().item())
                        best_path = fus_node_candidate_paths[best_path_idx]

                        sampled_fus_cc_paths_by_k[k][s_idx, cc_idx, num_selected_paths, : k + 1] = torch.tensor(
                            [x + 1 for x in best_path], dtype=torch.long, device=sampled_fus_cc_paths_by_k[k].device)
                        num_selected_paths += 1
                        used_fus_node_pairs.add((fus_id, cc_node_id))  

                    # Reinforcement sampling step: fill remaining capacity with unused high-scoring pairs
                    if num_selected_paths < max_paths_per_cc:
                        unused_fus_node_pairs = [
                            (fus_node_reach_score, fus_id, cc_node_id, fus_node_candidate_paths)
                            for fus_node_candidates in fus_cc_candidates
                            for (fus_node_reach_score, fus_id, cc_node_id, fus_node_candidate_paths) in fus_node_candidates
                            if (fus_id, cc_node_id) not in used_fus_node_pairs]

                        unused_fus_node_pairs.sort(key=lambda candidate: candidate[0], reverse=True)

                        max_extra_paths_per_cc_node = max(1, math.ceil((max_paths_per_cc - num_selected_paths) / max(1, len(fus_cc_candidates))))
                        cc_node_extra_path_counts = defaultdict(int)

                        for fus_node_reach_score, fus_id, cc_node_id, fus_node_candidate_paths in unused_fus_node_pairs:
                            if num_selected_paths >= max_paths_per_cc:
                                break
                       
                            if cc_node_extra_path_counts[cc_node_id] >= max_extra_paths_per_cc_node:
                                continue

                            path_pos_influence_scores = self.compute_path_pos_influence_scores((torch.tensor(fus_node_candidate_paths, dtype=torch.long).add_(1).unsqueeze(0).unsqueeze(0)), k)
                            path_influence_scores = path_pos_influence_scores.mean(dim=-1).view(-1)
                            best_path_idx = int(path_influence_scores.argmax().item())
                            best_path = fus_node_candidate_paths[best_path_idx]
                        
                            sampled_fus_cc_paths_by_k[k][s_idx, cc_idx, num_selected_paths, : k + 1] = torch.tensor(
                                [x + 1 for x in best_path], dtype=torch.long, device=sampled_fus_cc_paths_by_k[k].device)
                            num_selected_paths += 1
                            cc_node_extra_path_counts[cc_node_id] += 1
                            used_fus_node_pairs.add((fus_id, cc_node_id))

        # Trim padded paths per k to the maximum number used across all CCs
        for k in used_path_lengths:
            k_paths = sampled_fus_cc_paths_by_k[k]  
            max_cc_paths = int(((k_paths != config.PAD_VALUE).any(dim=-1).sum(dim=-1)).max().item())
            sampled_fus_cc_paths_by_k[k] = k_paths[:, :, :max_cc_paths, :] if max_cc_paths > 0 else k_paths[:, :, :0, :]

        torch.save(sampled_fus_cc_paths_by_k, file_name)
        
        return sampled_fus_cc_paths_by_k

    
    def get_fus_cc_sampled_paths(self, split):
        
        # Load sampled fusion-to-CC paths, or sample them if they do not exist for each component of the subgraph
        train_fus_cc_paths_path = self.fusion_diffusion_data_path / ("train_fus_cc_paths_" + "max_path_length_" + str(self.max_path_length) + 
                                                                  "_" + "max_cc_paths_" + str(self.max_paths_per_cc) + ".ph")
        val_fus_cc_paths_path = self.fusion_diffusion_data_path / ("val_fus_cc_paths_" + "max_path_length_" + str(self.max_path_length) + 
                                                                  "_" + "max_cc_paths_" + str(self.max_paths_per_cc) + ".ph")
        test_fus_cc_paths_path = self.fusion_diffusion_data_path / ("test_fus_cc_paths_" + "max_path_length_" + str(self.max_path_length) + 
                                                                  "_" + "max_cc_paths_" + str(self.max_paths_per_cc) + ".ph")
        if split == 'train_val':
            if train_fus_cc_paths_path.exists():
                print("...Loading sampled train fusion-to-CC paths...")
                self.train_fus_cc_paths = torch.load(train_fus_cc_paths_path)
            else: 
                print("...Sampling train fusion-to-CC paths...")
                self.train_fus_cc_paths = self.sample_fus_cc_paths_by_k(train_fus_cc_paths_path, self.train_cc_node_ids, self.train_subgraph_fus_set, self.fusion_diffusion_data_path,
                                        self.max_paths_per_cc, self.used_path_lengths)  

            if val_fus_cc_paths_path.exists():
               print("...Loading sampled validation fusion-to-CC paths...")
               self.val_fus_cc_paths = torch.load(val_fus_cc_paths_path)
            else:
                print("...Sampling validation fusion-to-CC paths...")
                self.val_fus_cc_paths = self.sample_fus_cc_paths_by_k(val_fus_cc_paths_path, self.val_cc_node_ids, self.val_subgraph_fus_set, self.fusion_diffusion_data_path,
                                        self.max_paths_per_cc, self.used_path_lengths)
       
        elif split == 'test':
            if test_fus_cc_paths_path.exists():
                print("...Loading sampled test fusion-to-CC paths...")
                self.test_fus_cc_paths = torch.load(test_fus_cc_paths_path)
            else:
                print("...Sampling test fusion-to-CC paths...")
                self.test_fus_cc_paths = self.sample_fus_cc_paths_by_k(test_fus_cc_paths_path, self.test_cc_node_ids, self.test_subgraph_fus_set, self.fusion_diffusion_data_path,
                                        self.max_paths_per_cc, self.used_path_lengths) 
      


# =====================================================
# Compute path position-wise influence scores
# =====================================================
         
    def compute_path_pos_influence_scores(self, paths, path_length):
        """
        Compute position-wise influence scores for sampled paths.
        
        Args:
            paths (Tensor): Tensor of shape [num_subgraphs, num_cc, num_paths, path_length].
            path_length (int): Length of the paths.

        Returns:
            Tensor: Position-wise influence scores for sampled paths of shap [num_subgraphs, num_cc, num_paths, path_length].
        """
        
        # Smoothing parameters for combining forward and backward scores
        tau = self.hyperparameters['tau']
        rho = tau / (1.0 - 2.0 * tau)
        eps = 1e-12
        num_positions = paths.size(-1)
        
        # Fusion source and component endpoint IDs for each path
        fus_ids = paths[..., 0].long()  
        cc_endpoint_ids = paths[..., -1].long()  

        # Initialize forward and backward scores
        forward_scores = torch.zeros_like(paths, dtype=torch.float32)
        backward_scores = torch.zeros_like(paths, dtype=torch.float32)

        # Forward scores: influence reception from fusion source to position t
        for path_pos in range(1, num_positions):
            forward_reach = self.weighted_diff_reach[path_pos]
            position_node_ids = paths[..., path_pos].long()
            forward_scores[..., path_pos] = forward_reach[fus_ids, position_node_ids]

        # Backward scores: propagation from position t to component endpoint
        for path_pos in range(num_positions):
            remaining_length = path_length - path_pos
            if remaining_length == 0:
                continue
            backward_reach = self.weighted_diff_reach[remaining_length]
            position_node_ids = paths[..., path_pos].long()
            backward_scores[..., path_pos] = backward_reach[position_node_ids, cc_endpoint_ids]

        # Path position-wise influence scores
        fwd_bwd_sum = forward_scores + backward_scores
        harmonic_mean = (2.0 * forward_scores * backward_scores) / fwd_bwd_sum.clamp_min(eps)
        path_pos_influence_scores = (harmonic_mean + rho * fwd_bwd_sum) / (1.0 + 2.0 * rho)
        path_pos_influence_scores = torch.where(fwd_bwd_sum > 0, path_pos_influence_scores, torch.zeros_like(path_pos_influence_scores))

        return path_pos_influence_scores 

     
    def compute_split_path_pos_influence_scores(self, split, fus_cc_paths):
        """
        Compute path position-wise influence scores for one data split.
        """
        
        fus_cc_path_pos_influence_scores: Dict[int, torch.Tensor] = {}

        for k in self.used_path_lengths:
            paths = fus_cc_paths[k]  
            fus_cc_path_pos_influence_scores[k] = self.compute_path_pos_influence_scores(paths, k)

        setattr(self, f"{split}_fus_cc_path_pos_influence_scores", fus_cc_path_pos_influence_scores)

        

# =====================================================
# Prepare data
# =====================================================

    def prepare_test_data(self):
        """
        Prepare test data by:
            - Initializing connected components for test subgraphs.
            - Computing fusion gene sets for test subgraphs.
            - Sampling fusion-to-CC paths for test subgraphs.
            - Computing path position-wise influence scores.
        """

        print("...Preparing test dataset...")

        # Initialize connected components for test subgraphs
        print("...Initializing connected components for test subgraphs...")
        self.test_cc_node_ids = self.initialize_subgraph_ccs(self.test_subgraphs)
        print("...Finished initializing connected components for test subgraphs...")

        # Compute fusion gene sets for test subgraphs
        print("...Getting fusion sets for test subgraphs...")
        self.get_subgraph_fus_sets(split='test')
        print("...Finished getting fusion sets for test subgraphs...")

        # Sample fusion-to-CC paths for test subgraphs
        print("...Sampling fusion-to-CC paths for test subgraphs...")
        self.get_fus_cc_sampled_paths(split='test')
        print("...Finished sampling fusion-to-CC paths for test subgraphs...")
        
        # Compute path position-wise influence scores for sampled test paths
        print("...Computing path position-wise influence scores for sampled test paths...")
        self.compute_split_path_pos_influence_scores("test", self.test_fus_cc_paths)
        print("...Finished computing path position-wise influence scores for sampled test paths...")
        
        print("...Preparation for test dataset completed...")
        
        
    def prepare_data(self):
        """
        Prepare training and validation data by:
            - Initializing connected components for train and validation subgraphs.
            - Computing fusion gene sets for train and validation subgraphs.
            - Computing weighted diffusion reachability matrices.
            - Sampling fusion-to-CC paths for train and validation subgraphs.
            - Computing path position-wise influence scores.
        """
       
        print("...Preparing train and validation datasets...", flush=True)
        
        # Initialize connected components for train and validation subgraphs
        print("...Initializing connected components for train and validation subgraphs...", flush=True)
        self.train_cc_node_ids = self.initialize_subgraph_ccs(self.train_subgraphs)
        self.val_cc_node_ids = self.initialize_subgraph_ccs(self.val_subgraphs)
        print("...Finished initializing connected components for train and validation subgraphs...", flush=True)
        
        # Compute fusion gene sets for train and validation subgraphs
        print("...Getting fusion sets for train and validation subgraphs...", flush=True)
        self.get_subgraph_fus_sets(split='train_val')
        print("...Finished getting fusion sets for train and validation subgraphs...", flush=True)

        # Compute weighted diffusion reachability matrices
        print("...Computing weighted diffusion reachability matrices...", flush=True)
        self.get_graph_diffusion_reachability()
        print("...Finished computing weighted diffusion reachability matrices...", flush=True)

        # Sample fusion-to-CC paths for train and validation subgraphs
        print("...Sampling fusion-to-CC paths for train and validation subgraphs...", flush=True)
        self.get_fus_cc_sampled_paths(split='train_val')
        print("...Finished sampling fusion-to-CC paths for train and validation subgraphs...", flush=True)

        # Compute path position-wise influence scores for sampled training paths
        print("...Computing path position-wise influence scores for sampled training paths...", flush=True)
        self.compute_split_path_pos_influence_scores("train", self.train_fus_cc_paths)
        print("...Finished computing path position-wise influence scores for sampled training paths...", flush=True)

        # Compute path position-wise influence scores for sampled validation paths
        print("...Computing path position-wise influence scores for sampled validation paths...", flush=True)
        self.compute_split_path_pos_influence_scores("val", self.val_fus_cc_paths)
        print("...Finished computing path position-wise influence scores for sampled validation paths...", flush=True)
        
        print("...Preparation for train and validation datasets completed...", flush=True)


        
# =====================================================
# Load and collate Batch Data
# =====================================================
    def _pad_collate(self, batch):
        """
        Collate function for batching disease pairs and their subgraph-level inputs.
        """
        
        batch_subgraphs_data = {}
        batch_subgraph_idx = []
        batch_cc_node_ids = []
        batch_fus_cc_paths = []
        batch_fus_cc_path_pos_influence_scores = []

        disease_pair, disease_pair_label, \
        subgraph_a_idx, subgraph_a_cc_node_ids, subgraph_a_fus_cc_paths, subgraph_a_fus_cc_path_pos_influence_scores, \
        subgraph_b_idx, subgraph_b_cc_node_ids, subgraph_b_fus_cc_paths, subgraph_b_fus_cc_path_pos_influence_scores = zip(*batch)

        # Collect unique subgraph-level data across the batch
        for pair_idx in range(len(disease_pair)):
            subgraph_a_idx_value = subgraph_a_idx[pair_idx].item()
            subgraph_b_idx_value = subgraph_b_idx[pair_idx].item()
            
            if subgraph_a_idx_value not in batch_subgraphs_data:
                    batch_subgraphs_data[subgraph_a_idx_value] = {'subgraph_idx': subgraph_a_idx[pair_idx],
                                                                  'cc_node_ids': subgraph_a_cc_node_ids[pair_idx],
                                                                  'fus_cc_paths': subgraph_a_fus_cc_paths[pair_idx],
                                                                  'fus_cc_path_pos_influence_scores': subgraph_a_fus_cc_path_pos_influence_scores[pair_idx]}

            if subgraph_b_idx_value not in batch_subgraphs_data:
                    batch_subgraphs_data[subgraph_b_idx_value] = {'subgraph_idx': subgraph_b_idx[pair_idx],
                                                                  'cc_node_ids': subgraph_b_cc_node_ids[pair_idx],
                                                                  'fus_cc_paths': subgraph_b_fus_cc_paths[pair_idx],
                                                                  'fus_cc_path_pos_influence_scores': subgraph_b_fus_cc_path_pos_influence_scores[pair_idx]}
                    
        # Reorganize unique subgraph data into batched lists
        for _, subgraph_data in sorted(batch_subgraphs_data.items()):
            batch_subgraph_idx.append(subgraph_data['subgraph_idx'])
            batch_cc_node_ids.append(subgraph_data['cc_node_ids'])
            batch_fus_cc_paths.append(subgraph_data['fus_cc_paths'])
            batch_fus_cc_path_pos_influence_scores.append(subgraph_data['fus_cc_path_pos_influence_scores'])

        # Stack disease pairs and labels
        batch_disease_pairs = torch.stack(disease_pair)
        batch_disease_pair_labels = torch.stack(disease_pair_label)
       
        # Stack subgraph indices 
        batch_subgraph_idx = torch.stack(batch_subgraph_idx)
        
        # Trim connected component node IDs to the maximum used length in the batch
        batch_cc_node_ids = torch.stack(batch_cc_node_ids)
        batch_size, max_num_cc, _ = batch_cc_node_ids.shape
        batch_cc_node_ids_reshaped = batch_cc_node_ids.view(batch_size * max_num_cc, -1)
        used_node_positions = (torch.sum(torch.abs(batch_cc_node_ids_reshaped), dim=0) != 0)
        batch_cc_node_ids = batch_cc_node_ids_reshaped[:, used_node_positions].view(batch_size, max_num_cc, -1)
        
        # Stack fusion-to-CC paths and path position-wise influence scores
        stacked_fus_cc_paths = {}
        stacked_fus_cc_path_pos_influence_scores = {} 
        for path_length in batch_fus_cc_paths[0].keys():
            stacked_fus_cc_paths[path_length] = torch.stack([subgraph_fus_paths[path_length] for subgraph_fus_paths in batch_fus_cc_paths], dim=0)  
            stacked_fus_cc_path_pos_influence_scores[path_length] = torch.stack([subgraph_fus_scores[path_length] for subgraph_fus_scores in batch_fus_cc_path_pos_influence_scores], dim=0)
       
        return {'disease_pairs': batch_disease_pairs,
                'disease_pair_labels': batch_disease_pair_labels,
                'subgraph_idx': batch_subgraph_idx,
                'cc_node_ids': batch_cc_node_ids,
                'fus_cc_paths': stacked_fus_cc_paths,
                'fus_cc_path_pos_influence_scores': stacked_fus_cc_path_pos_influence_scores}

    
    def train_dataloader(self):
        """
        Return a DataLoader for the train dataset.
        """
 
        print("...Initializing train dataloader...")
        
        dataset = DiseasePairDataset(self.train_disease_pairs, self.train_disease_pair_labels, self.train_cc_node_ids, self.train_fus_cc_paths, self.train_fus_cc_path_pos_influence_scores)
                                
        loader = DataLoader(dataset, batch_size=self.hyperparameters['batch_size'], shuffle=True, collate_fn=self._pad_collate)
  
        return loader


    def val_dataloader(self):
        """
        Return a DataLoader for the validation dataset.
        """
  
        print("...Initializing validation dataloader...")

        dataset = DiseasePairDataset(self.val_disease_pairs, self.val_disease_pair_labels, self.val_cc_node_ids, self.val_fus_cc_paths, self.val_fus_cc_path_pos_influence_scores)
        
        loader = DataLoader(dataset, batch_size=self.hyperparameters['batch_size'], shuffle=False, collate_fn=self._pad_collate, drop_last=False)
        
        return loader


    def test_dataloader(self):
        """
        Return a DataLoader for the test dataset.
        """
        
        self.prepare_test_data()
        
        print("...Initializing test dataloader...")
     
        dataset = DiseasePairDataset(self.test_disease_pairs, self.test_disease_pair_labels, self.test_cc_node_ids, self.test_fus_cc_paths, self.test_fus_cc_path_pos_influence_scores)
                                  
        loader = DataLoader(dataset, batch_size=self.hyperparameters['batch_size'], shuffle=False, collate_fn=self._pad_collate)
        
        return loader



# =====================================================
# Configure optimizer and scheduler
# =====================================================

    def configure_optimizers(self):
        """
        Configure AdamW optimizer and OneCycleLR scheduler.
        """
        
        learning_rate = self.hyperparameters['learning_rate']
        weight_decay = self.hyperparameters['weight_decay']
        max_learning_rate = self.hyperparameters['max_lr']

        # Split parameters into weight-decay and no-weight-decay groups
        no_weight_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight", "bn.weight", "norm.weight")
        weight_decay_params = []
        no_weight_decay_params = []
        for param_name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(term in param_name for term in no_weight_decay_terms):
                no_weight_decay_params.append(param)
            else:
                weight_decay_params.append(param)
                
        # AdamW optimizer
        optimizer = torch.optim.AdamW([{"params": weight_decay_params, "weight_decay": weight_decay},
                                       {"params": no_weight_decay_params, "weight_decay": 0.0}],
                                       lr=learning_rate,
                                       betas=(0.9, 0.999),
                                       eps=1e-8)

        # OneCycleLR schedule
        total_steps = self.trainer.estimated_stepping_batches
        div_factor = max(max_learning_rate / max(learning_rate, 1e-12), 1.0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=max_learning_rate,
                                                        total_steps=total_steps,
                                                        pct_start=0.03,            
                                                        anneal_strategy="cos",      
                                                        div_factor=div_factor,
                                                        final_div_factor=1e2)

        return {"optimizer": optimizer,
                "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"}}


      


