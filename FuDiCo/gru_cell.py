# PyTorch Imports
import torch
import torch.nn as nn



class GRUCell(nn.Module):
    """
    Fusion influence-aware GRU cell.
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Define update gate parameters 
        self.W_z = nn.Linear(self.input_size, self.hidden_size)
        self.U_z = nn.Linear(self.hidden_size, self.hidden_size)
        self.V_z = nn.Linear(1, self.hidden_size)

        # Define reset gate parameters
        self.W_r = nn.Linear(self.input_size, self.hidden_size)
        self.U_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.V_r = nn.Linear(1, self.hidden_size) 

        # Define candidate state parameters
        self.W_n = nn.Linear(self.input_size, self.hidden_size)
        self.U_n = nn.Linear(self.hidden_size, self.hidden_size) 
        self.V_n = nn.Linear(1, self.hidden_size)  

 
    def forward(self, node_embedding, prev_hidden_state, pos_influence_score, accumulated_influence_score):
        """
        Compute one fusion influence-aware GRU update for a path position.
        """
        
        # Update gate conditioned on position-wise influence score
        update_gate = torch.sigmoid(self.W_z(node_embedding) + self.U_z(prev_hidden_state) + self.V_z(pos_influence_score))

        # Reset gate conditioned on accumulated influence score
        reset_gate = torch.sigmoid(self.W_r(node_embedding) + self.U_r(prev_hidden_state) + self.V_r(accumulated_influence_score))

        # Candidate state conditioned on position-wise influence score
        candidate_state = torch.tanh(self.W_n(node_embedding) + self.U_n(reset_gate * prev_hidden_state) + self.V_n(pos_influence_score))
        
        hidden_state = (1 - update_gate) * prev_hidden_state + update_gate * candidate_state
        
        return hidden_state
        
