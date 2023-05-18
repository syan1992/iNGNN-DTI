import os
import numpy as np
import pdb
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn import preprocessing
import networkx as nx

from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch_geometric.utils as utils
import torch

import copy
MAX_SEQ_PROTEIN = 1024 
MAX_SEQ_DRUG = 128

# '?' padding
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z', '[CLS]', '[SEP]', '[PAD]']


smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y','@', '[CLS]', '[SEP]','[PAD]']

pl = preprocessing.LabelEncoder()
dl = preprocessing.LabelEncoder()
enc_protein_l = pl.fit(amino_char)
enc_drug_l = dl.fit(smiles_char)

def label_smiles(x):
        return enc_drug_l.transform(x).T

def label_sequence(x):
        return enc_protein_l.transform(x).T

def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	ll = len(temp)
	if len(temp) < MAX_SEQ_PROTEIN:
		l = len(temp)
		input_mask = ([1] * l) + ([0] * (MAX_SEQ_PROTEIN - l))	
		temp = temp + ['[PAD]'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		l = MAX_SEQ_PROTEIN
		input_mask = ([1] * l)
		temp = temp [:MAX_SEQ_PROTEIN]

	return temp, np.asarray(input_mask)

def trans_drug(x):
    temp = list(x)
    temp = [i if i in smiles_char else '?' for i in temp]
    
    if len(temp) < MAX_SEQ_DRUG:
        l = len(temp)
        input_mask = ([1] * l) + ([0] * (MAX_SEQ_DRUG - l))	
        temp = temp + ['[PAD]'] * (MAX_SEQ_DRUG-len(temp))
    else:
        l = MAX_SEQ_DRUG
        input_mask = ([1] * l)
        temp = temp [:MAX_SEQ_DRUG]
    return temp, np.asarray(input_mask)

# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_structure_table = {'A': 1, 'C': 5, 'D': 4, 'E': 4, 'F': 6, 'G': 1, 'H': 2,
                    'I': 1, 'K': 2, 'L': 1, 'M': 5, 'N': 3, 'P': 7, 'Q': 3,
                    'R': 2, 'S': 8, 'T': 8, 'V': 1, 'W': 6, 'Y': 8}

res_polar_table = {'A': 1, 'C': 3, 'D': 4, 'E': 4, 'F': 1, 'G': 3, 'H': 2,
                    'I': 1, 'K': 2, 'L': 1, 'M': 1, 'N': 3, 'P': 1, 'Q': 3,
                    'R': 2, 'S': 3, 'T': 3, 'V': 1, 'W': 1, 'Y': 3}

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

res_polar_table = dic_normalize(res_polar_table)
res_structure_table = dic_normalize(res_structure_table)
# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def residue_features(residue):

    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)

    #return np.array([res_structure_table[residue]]+[res_polar_table[residue]]+res_property1+res_property2)

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(amino_char)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], amino_char)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

# target feature for target graph
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(amino_char), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in amino_char:
                    count += 1
                    continue
                pfm_mat[amino_char.index(res), count] += 1
                count += 1

    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    return pssm_mat

def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

# target aln file save in data/dataset/aln
def target_to_feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)
    return feature

# pconsc4 predicted contact map save in data/dataset/pconsc4
def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map <8)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    target_edge_index = np.array(target_edge_index)
    return target_size, target_feature, target_edge_index

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])

    return c_size, features, edge_index

def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))

def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB
# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/home/yan/code/DTI/datasetDTI', dataset='DAVIS', setnum=0, phase='train', dataFrame=None, pre_transform=None, transform=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.msa_path = '/home/yan/code/DTI/datasetDTI/' + dataset + '/aln'
        self.contact_path = '/home/yan/code/DTI/datasetDTI/'+dataset+'/alphafold'
        self.protein_feature = np.load('/home/yan/code/DTI/datasetDTI/unsup_feature/davis_protein_feature.npy', allow_pickle=True).item()
        self.drug_feature = np.load('/home/yan/code/DTI/datasetDTI/unsup_feature/davis_smiles_feature.npy', allow_pickle=True).item()
        self.df = dataFrame
        self.setnum = setnum
        self.phase = phase
        self.k_hop = 2
        if os.path.exists(self.processed_paths[0]):
            self.data_mol = torch.load(self.processed_paths[0])
            self.data_pro = torch.load(self.processed_paths[1])
        else:
            self.process()
    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['DAVIS_unseen_prot/'+self.dataset + '_' + str(self.setnum) + '_' +self.phase + '_data_mol.pt', 'DAVIS_unseen_prot/'+self.dataset + '_' + str(self.setnum) + '_' +self.phase + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        print('processing......')
        data_list_mol = []
        data_list_pro = []

        data_len = len(self.df)

        data_smiles = {}
        data_target = {}
        for i in range(data_len):
            print(i)
            smiles = self.df['SMILES'][i]
            tar_key = self.df['target_id'][i]#self.df['Gene'][i]
            labels = self.df['y'][i]#self.df['Label'][i]
            tar_seq = self.df['target_seq'][i]#self.df['Target Sequence'][i]

            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_to_graph(smiles)
            target_size, target_features, target_edge_index = target_to_graph(tar_key, tar_seq, self.contact_path, self.msa_path)
            drug_node_indices = []
            drug_edge_indices = []
            drug_indicators = []
            drug_edge_index_start = 0
            for node_idx in range(c_size):
                    drug_sub_nodes, drug_sub_edge_index, _, drug_edge_mask = utils.k_hop_subgraph(
                            node_idx, 
                            self.k_hop, 
                            torch.tensor(edge_index).T,
                            relabel_nodes=True, 
                            num_nodes = c_size
                           )
                    drug_node_indices.append(drug_sub_nodes)
                    drug_edge_indices.append(drug_sub_edge_index + drug_edge_index_start)
                    drug_indicators.append(torch.zeros(drug_sub_nodes.shape[0]).fill_(node_idx))
                    drug_edge_index_start += len(drug_sub_nodes)            

            prot_node_indices = []
            prot_edge_indices = []
            prot_indicators = []
            prot_edge_index_start = 0
            for node_idx in range(target_size):
                    prot_sub_nodes, prot_sub_edge_index, _, prot_edge_mask = utils.k_hop_subgraph(
                            node_idx,
                            self.k_hop,
                            torch.tensor(target_edge_index).T,
                            relabel_nodes=True,
                            num_nodes = target_size
                           )
                    prot_node_indices.append(prot_sub_nodes)
                    prot_edge_indices.append(prot_sub_edge_index + prot_edge_index_start)
                    prot_indicators.append(torch.zeros(prot_sub_nodes.shape[0]).fill_(node_idx))
                    prot_edge_index_start += len(prot_sub_nodes)
            
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]), d_feature=self.drug_feature[smiles], subgraph_node_index=torch.cat(drug_node_indices), subgraph_edge_index=torch.cat(drug_edge_indices, dim=1), subgraph_indicator_index=torch.cat(drug_indicators))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

            p = str(tar_seq[0:1022])
            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]),key=tar_key, p_feature=self.protein_feature[p], subgraph_node_index=torch.cat(prot_node_indices), subgraph_edge_index=torch.cat(prot_edge_indices, dim=1), subgraph_indicator_index=torch.cat(prot_indicators))
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
            
            #data_list_mol.append(GCNData_mol)
            #data_list_pro.append(GCNData_pro)
            
            data_smiles[smiles] = GCNData_mol
            data_target[tar_seq] = GCNData_pro

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
        self.data_mol = data_smiles#data_list_mol
        self.data_pro = data_target#data_list_pro

        torch.save(self.data_mol, self.processed_paths[0])
        torch.save(self.data_pro, self.processed_paths[1])

    def extract_subgraphs(self):
        print("Extracting {}-hop subgraphs...".format(self.k_hop))
        self.subgraph_node_index = []

        self.subgraph_edge_index = []

        # (i.e. which node in a graph)
        self.subgraph_indicator_index = []

        if self.use_subgraph_edge_attr:
            self.subgraph_edge_attr = []

        for i in range(len(self.dataset)):
            if self.cache_path is not None:
                filepath = "{}_{}.pt".format(self.cache_path, i)
                if os.path.exists(filepath):
                    continue
            graph = self.dataset[i]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicators = []
            edge_index_start = 0

            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                    node_idx, 
                    self.k_hop, 
                    graph.edge_index,
                    relabel_nodes=True, 
                    num_nodes=graph.num_nodes
                    )
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index + edge_index_start)
                indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask]) # CHECK THIS DIDN"T BREAK ANYTHING
                edge_index_start += len(sub_nodes)

            if self.cache_path is not None:
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    subgraph_edge_attr = torch.cat(edge_attributes)
                else:
                    subgraph_edge_attr = None
                torch.save({
                    'subgraph_node_index': torch.cat(node_indices),
                    'subgraph_edge_index': torch.cat(edge_indices, dim=1),
                    'subgraph_indicator_index': torch.cat(indicators).type(torch.LongTensor),
                    'subgraph_edge_attr': subgraph_edge_attr
                }, filepath)
            else:
                self.subgraph_node_index.append(torch.cat(node_indices))
                self.subgraph_edge_index.append(torch.cat(edge_indices, dim=1))
                self.subgraph_indicator_index.append(torch.cat(indicators))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    self.subgraph_edge_attr.append(torch.cat(edge_attributes))
        print("Done!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smiles = self.df['SMILES'][idx]
        tar_seq = self.df['target_seq'][idx]
        label = self.df['y'][idx]
        drug = copy.deepcopy(self.data_mol[smiles])
        prot = copy.deepcopy(self.data_pro[tar_seq])
        drug.__setitem__('y',torch.FloatTensor([label]))
        prot.__setitem__('y',torch.FloatTensor([label]))
        return drug, prot
