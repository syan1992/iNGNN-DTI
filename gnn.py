import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, BatchNorm, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj

import pdb
# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=32, num_features_mol=32, output_dim=128, dropout=0.2):
        super(GNNNet, self).__init__()

        print('GNNNet Loaded')
        self.n_output = n_output
 
        self.num_feature_pro = num_features_pro

        self.mol_conv1 = GATConv(78, num_features_mol, heads=4)
        self.mol_lin1 = torch.nn.Linear(78, 4 * num_features_mol)
        self.mol_conv2 = GATConv(num_features_mol * 4, num_features_mol, heads=4)
        self.mol_lin2 = torch.nn.Linear(4 * num_features_mol, 4 * num_features_mol)
        self.mol_conv3 = GATConv(
            4 * num_features_mol, output_dim, concat=False)
        self.mol_lin3 = torch.nn.Linear(4 * num_features_mol, output_dim)

        self.pro_conv1 = GATConv(70, num_features_pro, heads=4)
        self.lin1 = torch.nn.Linear(70, 4 * num_features_pro)
        self.pro_conv2 = GATConv(4 * num_features_pro, num_features_pro, heads=4)
        self.lin2 = torch.nn.Linear(4 * num_features_pro, 4 * num_features_pro)
        self.pro_conv3 = GATConv(
            4 * num_features_pro, output_dim, concat=False)
        self.lin3 = torch.nn.Linear(4 * num_features_pro, output_dim)

        self.relu = nn.ELU()
        self.bn1 = BatchNorm(128)
        self.bn2 = BatchNorm(128)
        self.bn3 = BatchNorm(128)
        self.bn1_drug = BatchNorm(128)
        self.bn2_drug = BatchNorm(128)
        self.bn3_drug = BatchNorm(128)
    def forward(self, data_drug, data_pro):
        bs = data_pro.batch[-1]+1
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_drug.x, data_drug.edge_index, data_drug.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch
         
        #drug
        x = self.bn1_drug(self.relu(self.mol_conv1(mol_x, mol_edge_index) + self.mol_lin1(mol_x)))
        x = self.bn2_drug(self.relu(self.mol_conv2(x, mol_edge_index) + self.mol_lin2(x)))
        x = self.bn3_drug(self.relu(self.mol_conv3(x, mol_edge_index) + self.mol_lin3(x)))
        x = gep(x, mol_batch)  # global pooling

	#protein
        xt = self.bn1(self.relu(self.pro_conv1(target_x, target_edge_index) + self.lin1(target_x)))
        xt = self.bn2(self.relu(self.pro_conv2(xt, target_edge_index) + self.lin2(xt)))
        xt = self.bn3(self.relu(self.pro_conv3(xt, target_edge_index) + self.lin3(xt)))
        xt = gep(xt, target_batch)  # global pooling

        return xt, x
