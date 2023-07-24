"""PyG Module for Virtual Node"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.norm import LayerNorm

"""
The code are adapted from
https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred
"""

#https://github.com/lucidrains/graph-transformer-pytorch/blob/db6b82efcead2959175a9c771346336eb75cd870/graph_transformer_pytorch/graph_transformer_pytorch.py#L19

class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args,**kwargs)

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x, res):
        #pdb.set_trace()
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)

class CatResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim * 2, dim, bias = False),
	    nn.ReLU()
        )

    def forward(self, x, res):
        #pdb.set_trace()
        gate_input = torch.cat((x, res), dim = -1)
        gate = self.proj(gate_input)
        return gate  

def FeedForward(dim, ff_mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        nn.ReLU(),
        nn.Linear(dim * ff_mult, dim)
    )

class AFTFull(nn.Module):
    def __init__(self, seq_len: int, d_model: int, has_bias: bool = True):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.pos_bias = nn.Parameter(torch.Tensor(seq_len, seq_len))

        self.activation = nn.Sigmoid()
        self.output = nn.Linear(d_model, d_model)
        #self.einsum = torch.einsum('ijb,jbd->ibd')

    def forward(self, query, key, mask=None):
        seq_len, _, _ = query.shape
        seq_len1, _, _ = key.shape

        query = self.query(query)
        key = self.key(key)
        value = self.value(key)

        pos_bias = self.pos_bias[:seq_len, :seq_len1]
        pos_bias = pos_bias.unsqueeze(-1)
        #if mask is not None:
        #    pos_bias = pos_bias.masked_fill(mask == 0, -1e9)

        max_key = key.max(dim=0, keepdims=True)[0]
        max_pos_bias = pos_bias.max(dim=1,  keepdims=True)[0]

        exp_key = torch.exp(key - max_key)
        exp_pos_bias = torch.exp(pos_bias - max_pos_bias)

        num = torch.einsum('ijb,jbd->ibd', exp_pos_bias, exp_key * value)
        den = torch.einsum('ijb,jbd->ibd', exp_pos_bias, exp_key)

        weight = num/den
        y = self.activation(query) * weight

        return self.output(y.permute(1,0,2))

class VirtualNode(nn.Module):
    r"""Virtual Node from [OGB Graph Property Prediction Examples](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol).
    It adds an virtual node to all nodes in the graph. This trick is helpful for **Graph Level Task**.
    Note:
        To use this trick, call `update_node_emb` at first, then call `update_vn_emb`.
    Examples:
        [VirtualNode (PyG)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/VirtualNode.ipynb)
    Args:
        in_feats (int): Feature size before conv layer.
        out_feats (int): Feature size after conv layer.
        dropout (float, optional): Dropout rate on virtual node embedding.
        residual (bool, optional): If True, use residual connection.
    """

    def __init__(self, in_feats, out_feats, dropout=0.5, residual=False):
        super(VirtualNode, self).__init__()
        self.dropout = dropout
        # Add residual connection or not
        self.residual = residual

        # Set the initial virtual node embedding to 0.
        self.vn_emb = nn.Embedding(1, in_feats)
        # nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        if in_feats == out_feats:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(in_feats, out_feats)

        self.W0 = nn.Linear(in_feats, out_feats)
        self.W1 = nn.Linear(in_feats, out_feats)
        self.W2 = nn.Linear(in_feats, out_feats)
        self.nl_bn = nn.Sequential(
                nn.BatchNorm1d(in_feats),
                nn.ReLU()
        )
         
        # MLP to transform virtual node at every layer
        self.mlp_vn = nn.Sequential(
            nn.Linear(out_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU()
        )
        self.mlp_vn1 = nn.Sequential(
            nn.Linear(out_feats*2, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU()
        )
        self.mlp_vn2 = nn.Sequential(
            nn.Linear(out_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU()
        )

        self.mlp_vn3 = nn.Sequential(
            nn.Linear(out_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU()
        )
        self.reset_parameters()

        self.layers = nn.ModuleList([])
        with_feedforwards = None  
        for _ in range(1):
            self.layers.append(nn.ModuleList([
                nn.ModuleList([
                    AFTFull(3000, out_feats),
                    CatResidual(out_feats),
                    GatedResidual(out_feats)
                ]),
                nn.ModuleList([
                    FeedForward(out_feats,1),
                    GatedResidual(out_feats)
                ]) if with_feedforwards else None
            ]))

        self.layers_unsup = nn.ModuleList([])
        for _ in range(1):
            self.layers_unsup.append(nn.ModuleList([
                nn.ModuleList([
                    AFTFull(3000, out_feats),
                    CatResidual(out_feats),
                    GatedResidual(out_feats)
                ]),
                nn.ModuleList([
                    FeedForward(out_feats,1),
                    GatedResidual(out_feats)
                ]) if with_feedforwards else None
            ]))

        self.ln1 = LayerNorm(in_feats)
        self.ln2 = LayerNorm(in_feats)
        self.bn2 = nn.BatchNorm1d(in_feats)

    def reset_parameters(self):
        if not isinstance(self.linear, nn.Identity):
            self.linear.reset_parameters()

        for c in self.mlp_vn.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.vn_emb.weight.data, 0)

    def update_node_emb(self, x, edge_index, batch, vx=None, sync=None, unsup=None):
        r""" Add message from virtual nodes to graph nodes.
        Args:
            x (torch.Tensor): The input node feature.
            edge_index (torch.LongTensor): Graph connectivity.
            batch (torch.LongTensor): Batch vector, which assigns each node to a specific example.
            vx (torch.Tensor, optional): Optional virtual node embedding.
        Returns:
            (torch.Tensor): The output node feature.
            (torch.Tensor): The output virtual node embedding.
        """
        # Virtual node embeddings for graphs
        if vx is not None:
            h = x+vx[batch]
        else:
            h = x
        if vx is None:
            vx = self.vn_emb(torch.zeros(
                batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        if sync is None:
            sync = torch.zeros(x.shape).to(edge_index.dtype).to(edge_index.device)
        return h, vx, sync

    def update_vn_emb(self, x, batch, vx, unsup=None, layer=0):
        r""" Add message from graph nodes to virtual node.
        Args:
            x (torch.Tensor): The input node feature.
            batch (LongTensor): Batch vector, which assigns each node to a specific example.
            vx (torch.Tensor): Optional virtual node embedding.
        Returns:
            (torch.Tensor): The output virtual node embedding.
        """
        vx = self.W0(vx)+self.W1(global_add_pool(x,batch))
        vx = self.nl_bn(vx)
        vx = F.dropout(vx, self.dropout, training=self.training)
        return vx

    def update_cross_feature(self, x, batch, sync=None, final=None):
        h = self.mlp_vn1(torch.cat((x,F.relu(sync)),dim=1))
        h = F.dropout(h, self.dropout, training=self.training)
        return h 

    def update_feature(self, x, batch, query, query_batch, value, query_unstruct, sync_last=None):
        x_dense, mask = to_dense_batch(x,batch)
        query_dense, query_mask = to_dense_batch(query, query_batch)
        value_dense, _ = to_dense_batch(value, batch)

        x_dense = self.ln2(x_dense)
        query_dense = self.ln2(query_dense)
        value_dense = self.ln2(value_dense)
        mask1 = torch.bmm(query_mask.unsqueeze(-1).float(), mask.unsqueeze(1).float())
        mask2 = torch.bmm(mask.unsqueeze(-1).float(), query_mask.unsqueeze(1).float())
        for attn_block, ff_block in self.layers:
            attn, _ ,attn_residual = attn_block
            feature = attn(query_dense.permute(1,0,2), x_dense.permute(1,0,2), mask1.permute(1,2,0))
            feature = feature[query_mask]+query
            #feature = attn_residual(feature[query_mask], query)#self.mlp_vn2(feature1[query_mask]+query)
            if ff_block is not None:
                ff, ff_residual = ff_block
                feature = ff(feature)+feature#ff_residual(ff(feature), feature)
        #layernorm is important
        feature = self.ln1(feature, query_batch)
        return feature 
    
    def update_feature_final(self, x, batch, query, unsup):
        x_dense, mask = to_dense_batch(x,batch)
        for attn_block, _ in self.layers:
            attn, _, _ = attn_block
            feature = attn(query.unsqueeze(1).permute(1,0,2), x_dense.permute(1,0,2))
            feature = feature.squeeze()+query
        feature = self.bn2(feature)        
        return feature
     
