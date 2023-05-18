"""PyG Module for Virtual Node"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch_geometric.nn as gnn
from torch_geometric.nn import global_add_pool, global_mean_pool 
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.norm import LayerNorm
import math
from einops import rearrange, reduce, repeat
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

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) #/ math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        #attention = F.sigmoid(scores)
        #mask_tmp = mask.sum(2)
        #mask = mask_tmp.masked_fill(mask_tmp==0, 1)
        #attention = attention / mask.unsqueeze(-1)
        return attention.matmul(value) 

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v1 = nn.Linear(in_features, in_features, bias)
        self.linear_v2 = nn.Linear(in_features, in_features, bias)
        self.linear_o1 = nn.Linear(in_features, in_features, bias)
        self.linear_o2 = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask1=None,mask2=None):
        q, k, v1, v2 = self.linear_q(q), self.linear_k(k), self.linear_v1(v), self.linear_v2(q)
        #if self.activation is not None:
        #    q = self.activation(q)
        #    k = self.activation(k)
        #    v1 = self.activation(v1)
        #    v2 = self.activation(v2)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v1 = self._reshape_to_batches(v1)
        v2 = self._reshape_to_batches(v2)

        if mask1 is not None:
            mask1 = mask1.repeat(self.head_num, 1, 1)
            mask2 = mask2.repeat(self.head_num, 1, 1)
        
        y1 = ScaledDotProductAttention()(q, k, v1, mask1)
        y1 = self._reshape_from_batches(y1)
        y1 = self.linear_o1(y1)

        y2 = ScaledDotProductAttention()(k,q,v2,mask2)
        y2 = self._reshape_from_batches(y2)
        y2 = self.linear_o2(y2)

        #if self.activation is not None:
        #    y1 = self.activation(y1)
        #    y2 = self.activation(y2)
        return y1

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

'''
# Attention-Free Layer. #
class AFTFull(torch.nn.Module):
    def __init__(self, d_model):
        super(AFTFull, self).__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.d_model = d_model
        
        self.wq = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(
            d_model, d_model, bias=False)
    
    def forward(self, q, k, v):
        pdb.set_trace()
        q_input = F.elu(self.wq(q)) + 1.0
        k_input = F.elu(self.wk(k)) + 1.0
        v_input = self.wv(v)

        # Prefix sums for causality. #
        kv_input  = torch.mul(k_input, v_input)
        k_prefix  = torch.cumsum(k_input, dim=1)
        kv_prefix = torch.cumsum(kv_input, dim=1)
        
        kv_softmax = torch.div(
            kv_input + kv_prefix, k_prefix)
        attn_outputs = torch.mul(q_input, kv_softmax)
        return attn_outputs
'''
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
'''
class AFTFull(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=128):
        super().__init__()
        
        #max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        #dim: the embedding dimension of the tokens
        #hidden_dim: the hidden dimension used inside AFT Full
        #Number of heads is 1 as done in the paper
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.to_v_1 = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.project_1 = nn.Linear(hidden_dim, dim)
        self.wbias = nn.Parameter(torch.Tensor(max_seqlen, max_seqlen))
        nn.init.xavier_uniform_(self.wbias)

    def forward(self, x, x1, mask=None):
        B, T, _ = x.shape
        B1, T1, _ = x1.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x1).view(B1, T1, self.hidden_dim)
        V = self.to_v(x1).view(B1, T1, self.hidden_dim)
        V_1 = self.to_v_1(x).view(B, T, self.hidden_dim)

        temp_wbias = self.wbias[:T, :T1] # sequences can still be variable length
        Q_sig = torch.sigmoid(Q)
        #Q_sig_1 = torch.sigmoid(K)

        temp = torch.exp(temp_wbias.unsqueeze(0)) @ torch.mul(torch.exp(K), V)
        #temp1 = torch.exp(temp_wbias.T.unsqueeze(0))@torch.mul(torch.exp(Q), V_1)
        temp_nominator = (torch.exp(temp_wbias.unsqueeze(0)) @ torch.exp(K))
        weighted = temp / temp_nominator
        #weighted1 = temp1 / (torch.exp(temp_wbias.T.unsqueeze(0))@torch.exp(Q))
        weighted = torch.where(torch.isnan(weighted), torch.full_like(weighted, 1e-9), weighted)

        Yt = torch.mul(Q_sig, weighted)
        #Yt1 = torch.mul(Q_sig_1, weighted1)
        #has_nan = torch.any(torch.isnan(Yt))
        #if has_nan:
        #       pdb.set_trace() 
        Yt = Yt.view(B, T, self.hidden_dim)
        #Yt1 = Yt1.view(B1, T1, self.hidden_dim)
        
        Yt = self.project(Yt)
        #has_nan = torch.any(torch.isnan(Yt))
        #if has_nan:
        #       pdb.set_trace()
        #Yt1 = self.project_1(Yt1)
        return Yt
'''
'''
class AFTSimple(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=128):
        super().__init__()
        
        #max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        #dim: the embedding dimension of the tokens
        #hidden_dim: the hidden dimension used inside AFT Full
        
        #Number of Heads is 1 as done in the paper.
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)

    def forward(self, x, x1):
        B, T, _ = x.shape
        B1, T1, _ = x1.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x1).view(B1, T1, self.hidden_dim)
        V = self.to_v(x1).view(B1, T1, self.hidden_dim)

        
        #From the paper
        
        weights = torch.mul(torch.softmax(K, 1), V).sum(dim=1, keepdim=True)
        Q_sig = torch.sigmoid(Q)
        Yt = torch.mul(Q_sig, weights)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt


class AFTFull(torch.nn.Module):
    def __init__(self, d_model):
        super(AFTFull, self).__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.d_model = d_model
        
        self.wq = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(
            d_model, d_model, bias=False)
    
    def forward(self, q, k):
        #q_input = self.sigmoid(self.wq(q))
        #k_input = torch.exp(self.wk(k))
        q_input = F.elu(self.wq(q)) + 1.0
        k_input = F.elu(self.wk(k)) + 1.0
        v_input = self.wv(k)

        # Prefix sums for causality. #
        kv_input  = torch.mul(k_input, v_input)
        k_prefix  = torch.cumsum(k_input, dim=1)
        kv_prefix = torch.cumsum(kv_input, dim=1)
        
        kv_softmax = torch.div(
            kv_input + kv_prefix, k_prefix)
        attn_outputs = torch.mul(q_input, kv_softmax)
        return attn_outputs
'''

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
        self.bn3 = nn.BatchNorm1d(in_feats)
        self.bn4 = nn.BatchNorm1d(in_feats)
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
        
        '''
        # Add message from graph nodes to virtual nodes
        vx = self.linear(vx)
        vx_temp = global_add_pool(x, batch) + vx

        # transform virtual nodes using MLP
        vx_temp = self.mlp_vn(vx_temp)

        if self.residual:
            vx = vx + F.dropout(
                vx_temp, self.dropout, training=self.training)
        else:
            vx = F.dropout(
                vx_temp, self.dropout, training=self.training)
        '''
        vx = self.W0(vx)+self.W1(global_add_pool(x,batch))
        vx = self.nl_bn(vx)
        vx = F.dropout(vx, self.dropout, training=self.training)
        return vx

    def Attention(self, query, context):
        weight = torch.matmul(query, torch.transpose(context, 2, 1))
        weight = torch.softmax(weight, dim=2)
        return weight

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
        for attn_block, ff_block in self.layers:
            attn, _, attn_residual = attn_block
            feature = attn(query.unsqueeze(1).permute(1,0,2), x_dense.permute(1,0,2))
            feature = feature.squeeze()+query
        feature = self.bn2(feature)        
        return feature
     
