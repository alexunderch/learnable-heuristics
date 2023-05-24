#!/usr/bin/env python
"""Generic network class for supervised regression
Created on: March 25, 2017
Author: Mohak Bhardwaj"""

import os 
import sys
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn.models import DeepGCNLayer
from torch_geometric.nn import GATv2Conv, GENConv, SoftmaxAggregation, PowerMeanAggregation


torch.autograd.set_detect_anomaly(True)


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dims, out_dim):
        super().__init__()

        self.mlp = nn.Sequential()
        dims = [input_dim] + hid_dims + [out_dim]
        for i in range(len(dims)-1):
            self.mlp.add_module('lay_{}'.format(i),nn.Linear(in_features=dims[i], out_features=dims[i+1]))
            if i+2 < len(dims):
                self.mlp.add_module('act_{}'.format(i), nn.LeakyReLU())
    def reset_parameters(self):
        for _, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)


class DeeperGCN(nn.Module):
    def __init__(self, in_channels: int, 
                       hidden_channels: int, 
                       num_layers: int, 
                       conv_aggr: str = "softmax",
                       aggr = None):
        super().__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, 
                           aggr=conv_aggr,
                           t=1.0, 
                           learn_t=True, 
                           num_layers=1, 
                           norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU()

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        if self.aggr is not None:
            x = self.aggr(x)
        return x


class SupervisedRegressionNetwork(nn.Module):
  def __init__(self, params):
    super().__init__()    
    self.initialized=False
    self.output_size = params['output_size']
    self.input_size = params['input_size']
    self.node_dim = params['neighborhood_size']
    self.hidden_dim = params['hidden_dim']
    self.learning_rate = params['learning_rate']
    self.batch_size =  params['batch_size']
    self.training_epochs = params['training_epochs']
    self.display_step = params['display_step']
    self.seed_val = params['seed_val']
    self.graph_layer_name  = "GAT",
    self.conv_aggr  = "mean"
    self.input_shape = [self.input_size]
    
    if params['mode'] == "gpu":
      self.device = 'cuda'
    else:
      self.device = 'cpu'
    self.device = torch.device(self.device)
    self.memory = torch.zeros(1, self.node_dim+1, 64).to(self.device)
    hidden_dim = self.hidden_dim

    self.feature_proj = MLP(self.input_size, [128], hidden_dim)
    self.goal_proj = MLP(2, [128], hidden_dim)

    if self.graph_layer_name == "DeepGCN":
        self.graph_layer = DeeperGCN(hidden_dim, num_layers=1, conv_aggr=self.conv_aggr)
    elif self.graph_layer_name == "GAT":
        self.graph_layer = GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False)
    else:
        self.graph_layer = GENConv(hidden_dim, hidden_dim, aggr=self.conv_aggr)
    self.rnn = nn.GRU(input_size = hidden_dim, 
                      hidden_size = 64)

    self.heuristic_head = nn.Linear(hidden_dim + 64, self.output_size)

    self.optimizer = torch.optim.AdamW(self.to(self.device).parameters(), lr=1e-3)
    self.loss_fn = torch.nn.MSELoss()

  def initialize(self):
    if not self.initialized:
      self.to(self.device)
      # self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
      # self.loss_fn = torch.nn.MSELoss()
      self.initialized=True
      print('Network created and initialized')


  def forward(self, data, hidden=None):
      if hidden is not None:    
          self.memory = hidden
      x, edge_index = data.x, data.edge_index.long()
      goal_x = self.goal_proj(data.y)
      x = self.feature_proj(x)
      x = self.graph_layer(x, edge_index, None)
      o, hidden_n = self.rnn(x.unsqueeze(0), self.memory)
      self.memory = hidden_n.detach()
      return self.heuristic_head(torch.cat([o[:, -1], goal_x], -1)).squeeze(-1)

  def train(self, database):
    #Shuffle the database
    for epoch in range(self.training_epochs):
      self.optimizer.zero_grad()
      random.shuffle(database)
      avg_cost = 0.
      total_batch = int(len(database)/self.batch_size)
      for i in range(total_batch):
        batch_x, batch_y = self.get_next_batch(database, i)
        for ind, xx in enumerate(batch_x):
          loss = self.loss_fn(batch_y[ind].unsqueeze(0).detach(), self(xx.to(self.device)))
          loss.backward()
        avg_cost+= loss.item()/total_batch
      self.optimizer.step()

      print (f"epoch: {epoch+1} cost={np.sqrt(avg_cost)}")
    print('optimization finished!')
    return np.sqrt(avg_cost)

  @torch.no_grad()
  def get_heuristic(self, data):
    self.training = False
    heuristic = self.forward(data.to(self.device))
    # print(heuristic.shape)
    return heuristic.item()

  def save_params(self, file_name):
    #file_path = os.path.join(os.path.abspath('saved_data/saved_models'), file_name +'.ckpt')
    torch.save(self.state_dict(), file_name + ".pth")
    print("Model saved in file: %s" % file_name)

  def load_params(self, file_name):
    #file_path = os.path.join(os.path.abspath('saved_data/saved_models'), file_name +'.ckpt')
    self.load_state_dict(torch.load(file_name + ".pth"))
    print('Weights loaded from file %s'%file_name)

  def get_next_batch(self, database, i):
    batch = database[i*self.batch_size: (i+1)*self.batch_size]
    batch_x = [_[0] for _ in batch]
    batch_y = torch.Tensor([_[1] for _ in batch]).to(self.device)
    return batch_x, batch_y
  
  
  # def save_summaries(self, vars, iter_idx, train=True):
  #   print('Writing summaries')
  #   summary_str = self.sess.run(self.summary_ops, 
  #                               feed_dict = {self.episode_stats_vars[0]: vars[0],
  #                                            self.episode_stats_vars[1]: vars[1],
  #                                            self.episode_stats_vars[2]: vars[2],
  #                                            self.episode_stats_vars[3]: vars[3],
  #                                            self.episode_stats_vars[4]: vars[4]})
  #   if train:
  #     self.train_writer.add_summary(summary_str, iter_idx)
  #     self.train_writer.flush()
  #   else:
  #     self.test_writer.add_summary(summary_str, iter_idx)
  #     self.test_writer.flush()      

  # def build_summaries(self):
  #   # variable_summaries(episode_reward)
  #   episode_reward = tf.Variable(0.) 
  #   episode_expansions = tf.Variable(0.)
  #   episode_expansions_std = tf.Variable(0.) 
  #   episode_accuracy = tf.Variable(0.)
  #   num_unsolved = tf.Variable(0)
  #   episode_stats_vars = [episode_reward, episode_expansions, episode_expansions_std, episode_accuracy, num_unsolved]
  #   episode_stats_ops = [tf.summary.scalar("Rewards", episode_reward), tf.summary.scalar("Expansions(Task Loss)", episode_expansions),tf.summary.scalar("Std. Expansions", episode_expansions_std), tf.summary.scalar("RMS(Surrogate Loss)", episode_accuracy), tf.summary.scalar("Number of Unsolved Envs", num_unsolved)]
  #   return episode_stats_ops, episode_stats_vars

  



