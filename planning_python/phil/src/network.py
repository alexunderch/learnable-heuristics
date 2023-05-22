import os 
import numpy as np
import random

import torch
import torch.nn as nn
from torch_geometric.nn.models import DeepGCNLayer, GAT, NNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


def build_mlp(input_dim: int, 
              output_dim: int, 
              hidden_dim: int = 128,
              dropout_rate: float = .0):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(dropout_rate),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(dropout_rate),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(dropout_rate),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, output_dim)
    )

class HeuristicNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, graph_layer: str = "DeepGCN") -> None:
        super().__init__()

        self.feature_proj = build_mlp(input_dim=input_dim, hidden_dim=128,
                                output_dim=128, dropout_rate=0.1)
        if graph_layer == "DeepGCN":
            self.agg_net = DeepGCNLayer()
        elif graph_layer == "GAT":
            self.graph_layer = GAT()
        else: 
            self.graph_layer = NNConv()
        self.rnn = nn.GRU()

        self.heuristic_head = nn.Linear(211, 1)

        self._hidden = ...

    def forward(self, node, goal_node):
        pass