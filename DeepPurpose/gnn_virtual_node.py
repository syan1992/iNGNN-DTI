import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, global_max_pool, GCNConv, GINConv, GATConv
# import virtual node
#from gtrick.pyg import VirtualNode
from DeepPurpose.virtual_node import VirtualNode
import pdb
from torch_geometric.utils import to_dense_batch
from DeepPurpose.satlayer import *
from torch_geometric.nn.norm import LayerNorm
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.autograd import Variable
class EGNN(nn.Module):

    def __init__(self, hidden_channels=128, out_channels=128, num_layers=3,
                 dropout=0.2, conv_type='gcn', mol=False):

        super(EGNN, self).__init__()

        self.mol = mol
        self.num_layers = num_layers

        self.convs_drug = nn.ModuleList()
        self.bns_drug = nn.ModuleList()
        self.vns_drug = nn.ModuleList()

        self.convs_prot = nn.ModuleList()
        self.bns_prot = nn.ModuleList()
        self.vns_prot = nn.ModuleList()

        self.mlp_drug = nn.ModuleList()
        self.mlp_prot = nn.ModuleList()        

        self.drug_encoder = torch.nn.Linear(78, hidden_channels)
        self.prot_encoder = torch.nn.Linear(70, hidden_channels)

        self.d = torch.nn.Linear(256, hidden_channels)
        self.p = torch.nn.Linear(256, hidden_channels)
        for i in range(self.num_layers):
            if conv_type == 'gin':
                    self.convs_drug.append(
                        GINConv(hidden_channels, hidden_channels))
                    self.convs_prot.append(
                        GINConv(hidden_channels, hidden_channels))
            elif conv_type == 'gcn':
                    self.convs_drug.append(
                        GCNConv(hidden_channels, hidden_channels))
                    self.convs_prot.append(
                        GCNConv(hidden_channels, hidden_channels))
            elif conv_type == 'gat':
                    self.convs_drug.append(
                        GATConv(hidden_channels, hidden_channels))
                    self.convs_prot.append(
                        GATConv(hidden_channels, hidden_channels))
            elif conv_type=='sat':
                    self.convs_drug.append(KHopStructureExtractor(hidden_channels, gnn_type="gcn", num_layers=1))
                    self.convs_prot.append(KHopStructureExtractor(hidden_channels, gnn_type="gcn", num_layers=1))
            self.bns_drug.append(torch.nn.BatchNorm1d(hidden_channels))
            self.bns_prot.append(torch.nn.BatchNorm1d(hidden_channels))
            self.bns_drug.append(torch.nn.BatchNorm1d(hidden_channels))
            self.bns_prot.append(torch.nn.BatchNorm1d(hidden_channels))
            # add a virtual node layer
            self.vns_drug.append(VirtualNode(hidden_channels, hidden_channels, dropout=dropout))
            self.vns_prot.append(VirtualNode(hidden_channels, hidden_channels, dropout=dropout))

            self.mlp_drug.append(nn.Linear(hidden_channels, hidden_channels))
            self.mlp_prot.append(nn.Linear(hidden_channels, hidden_channels))  
        self.vns_drug.append(VirtualNode(hidden_channels, hidden_channels, dropout=dropout))
        self.vns_prot.append(VirtualNode(hidden_channels, hidden_channels, dropout=dropout))

        self.dropout = dropout

        self.ln = LayerNorm(hidden_channels) 
        self.out = nn.Linear(hidden_channels, out_channels)

        self.W0 = nn.Linear(hidden_channels, hidden_channels)

        self.vn_emb_drug = nn.Embedding(1, hidden_channels)
        self.vn_emb_prot = nn.Embedding(1, hidden_channels)
        self.nl_bn = nn.Sequential(
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.mlp_d = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )

        self.mlp_p = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )

        self.mlp_d1 = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
        )

        self.mlp_p1 = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
        )
       
        self.lstm_drug = nn.LSTM(hidden_channels, hidden_channels, bidirectional=True, batch_first=True, num_layers=2)
        self.lstm_prot = nn.LSTM(hidden_channels, hidden_channels, bidirectional=True, batch_first=True, num_layers=2)

    def reset_parameters(self):
        if self.mol:
            for emb in self.node_encoder.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            nn.init.xavier_uniform_(self.drug_encoder.weight.data)
            nn.init.xavier_uniform_(self.prot_encoder.weight.data)

        for i in range(self.num_layers):
            self.convs_drug[i].reset_parameters()
            self.bns_drug[i].reset_parameters()
            self.vns_drug[i].reset_parameters()

            self.convs_prot[i].reset_parameters()
            self.bns_prot[i].reset_parameters()
            self.vns_prot[i].reset_parameters()

            
        self.out.reset_parameters()


    def forward(self, data_drug, data_pro, d_feature, p_feature):
        # get graph input
        drug_x, drug_edge_index, drug_batch, drug_subgraph_node_index, drug_subgraph_edge_index, drug_subgraph_indicator_index = data_drug.x, data_drug.edge_index, data_drug.batch, data_drug.subgraph_node_index, data_drug.subgraph_edge_index, data_drug.subgraph_indicator_index
        # get protein input
        prot_x, prot_edge_index, prot_batch, prot_subgraph_node_index, prot_subgraph_edge_index, prot_subgraph_indicator_index = data_pro.x, data_pro.edge_index, data_pro.batch, data_pro.subgraph_node_index, data_pro.subgraph_edge_index, data_pro.subgraph_indicator_index
        
        h_drug = self.drug_encoder(drug_x)
        h_prot = self.prot_encoder(prot_x)

        vx_drug = None 
        vx_prot = None
        sync_drug = None
        sync_prot = None
        vx = None 

        drug_feature = []
        prot_feature = []

        #h_drug_init = self.mlp_d(h_drug)
        #h_prot_init = self.mlp_p(h_prot)
        #sync_drug = self.vns_prot[-1].update_feature(h_drug_init, drug_batch, h_prot_init, prot_batch, h_drug_init, h_prot_init)           
        #sync_prot = self.vns_drug[-1].update_feature(h_prot_init, prot_batch, h_drug_init, drug_batch, h_prot_init, h_drug_init)

        #sync_prot_last = self.vns_drug[-1].update_cross_feature(sync_prot, drug_batch, h_drug)
        #sync_drug_last = self.vns_prot[-1].update_cross_feature(sync_drug, prot_batch, h_prot)

        #drug_feature.append(sync_prot_last.unsqueeze(1))
        #prot_feature.append(sync_drug_last.unsqueeze(1)) 
        for i, conv in enumerate(zip(self.convs_drug[0:-1], self.convs_prot[0:-1])):
            # drug use virtual node to update node embedding
            h_drug, vx_drug, sync_prot = self.vns_drug[i].update_node_emb(h_drug, drug_edge_index, drug_batch, vx_drug, sync_prot, d_feature)
            h_drug = conv[0](h_drug, drug_edge_index)
            
            # prot use virtual node to update node embedding 
            h_prot, vx_prot, sync_drug = self.vns_prot[i].update_node_emb(h_prot, prot_edge_index, prot_batch, vx_prot, sync_drug, p_feature)
            h_prot = conv[1](h_prot, prot_edge_index)
            
            h_drug = self.bns_drug[i](h_drug)
            h_drug = F.relu(h_drug)
            h_drug = F.dropout(h_drug, p=self.dropout, training=self.training)

            h_prot = self.bns_prot[i](h_prot)
            h_prot = F.relu(h_prot)
            h_prot = F.dropout(h_prot, p=self.dropout, training=self.training)

            h_drug_struct = scatter_mean(h_drug[drug_subgraph_node_index], drug_subgraph_indicator_index.long(), dim=0) 
            h_prot_struct = scatter_mean(h_prot[prot_subgraph_node_index], prot_subgraph_indicator_index.long(), dim=0)

            vx_drug = self.vns_drug[i].update_vn_emb(h_drug_struct, drug_batch, vx_drug)
            vx_prot = self.vns_prot[i].update_vn_emb(h_prot_struct, prot_batch, vx_prot)

        if self.mol:
            h = self.convs[-1](h, edge_index, edge_attr)
            h = F.dropout(h, self.dropout, training=self.training)
        else:
            h_drug, vx_drug, sync_prot = self.vns_drug[-2].update_node_emb(h_drug, drug_edge_index, drug_batch, vx_drug, sync_prot, d_feature)
            h_drug = self.convs_drug[-1](h_drug, drug_edge_index)

            h_prot, vx_prot, sync_drug = self.vns_prot[-2].update_node_emb(h_prot, prot_edge_index, prot_batch, vx_prot, sync_drug, p_feature)
            h_prot = self.convs_prot[-1](h_prot, prot_edge_index)
             
            h_drug = self.bns_drug[-2](h_drug)
            h_drug = F.relu(h_drug)
            h_drug = F.dropout(h_drug, p=self.dropout, training=self.training)

            h_prot = self.bns_prot[-2](h_prot)
            h_prot = F.relu(h_prot)
            h_prot = F.dropout(h_prot, p=self.dropout, training=self.training)

            h_drug_struct = scatter_mean(h_drug[drug_subgraph_node_index], drug_subgraph_indicator_index.long(), dim=0)
            h_prot_struct = scatter_mean(h_prot[prot_subgraph_node_index], prot_subgraph_indicator_index.long(), dim=0)

            sync_drug = self.vns_prot[-2].update_feature(h_drug_struct, drug_batch, h_prot_struct, prot_batch, h_drug_struct, h_prot_struct, sync_drug)
            sync_prot = self.vns_drug[-2].update_feature(h_prot_struct, prot_batch, h_drug_struct, drug_batch, h_prot_struct, h_drug_struct, sync_prot)

            #sync_prot_last = self.vns_drug[-2].update_cross_feature(h_drug, drug_batch, sync_prot)
            #sync_drug_last = self.vns_prot[-2].update_cross_feature(h_prot, prot_batch, sync_drug)

            #drug_feature.append(sync_prot_last)#.unsqueeze(1))
            #prot_feature.append(sync_drug_last)#.unsqueeze(1))
            
            #sync_drug = self.vns_prot[-2].update_feature(sync_prot_last, drug_batch, sync_drug_last, prot_batch, sync_prot_last, sync_drug_last)
            #sync_prot = self.vns_drug[-2].update_feature(sync_drug_last, prot_batch, sync_prot_last, drug_batch, sync_drug_last, sync_prot_last)


        '''
        h_0_drug = Variable(torch.zeros(4, drug_feature.size(0), 128).cuda()) #hidden state
        c_0_drug = Variable(torch.zeros(4, drug_feature.size(0), 128).cuda()) #internal state

        h_0_prot = Variable(torch.zeros(4, prot_feature.size(0), 128).cuda()) #hidden state
        c_0_prot = Variable(torch.zeros(4, prot_feature.size(0), 128).cuda()) #internal state

        lstm_drug, (h_0_drug, c_0_drug) = self.lstm_drug(drug_feature, (h_0_drug, c_0_drug)) 
        lstm_prot, (h_0_prot, c_0_prot) = self.lstm_prot(prot_feature, (h_0_prot, c_0_prot))
        '''

        #sync_drug_unsup = self.vns_prot[-1].update_feature_final(h_prot_struct, prot_batch, p_feature, p_feature)
        #sync_prot_unsup = self.vns_drug[-1].update_feature_final(h_drug_struct, drug_batch, d_feature, d_feature)

        final_drug = self.mlp_d1(torch.cat((global_mean_pool(sync_prot, drug_batch), d_feature),dim=1))
        final_prot = self.mlp_p1(torch.cat((global_mean_pool(sync_drug, prot_batch), p_feature),dim=1))

        #final_drug = global_mean_pool(h_drug, drug_batch)
        #final_prot = global_mean_pool(h_prot, prot_batch)
        return final_drug, final_prot
