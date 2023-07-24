# Prediction of Drug - Target Interaction with Interpretable Nested Graph Neural Network and Pretrained Molecule Models (iNGNN-DTI)
## Introduction
![image](https://github.com/syan1992/iNGNN-DTI/blob/master/figure/archi.png)

iNGNN-DTI is a framework for the drug-target interaction prediction. The model extracts features of from the graph data of drugs and targets, employing a specific type of graph neural network known as the nested graph neural network (NGNN), in which the target graph is created using Alphafold2. We use the attention-free transformer (AFT) module to capture the interaction information between the substructures of drugs and targets. To improve the feature representations, we integrate features learned by models that are pre-trained on large unlabeled small molecule and protein datasets for the drugs and targets, respectively. 

## Environment
We conduct our experiments with python3.8. Here are the requirements
'''
descriptastorus
matplotlib
networkx
numpy
pandas
prettytable
rdkit
Requests
scikit_learn
scipy
subword_nmt
torch
torch_geometric
torchvision
'''

## Usage

'''
python main.py
'''

## Acknowledgement
DeepPurpose: https://github.com/kexinhuang12345/DeepPurpose
Nested Graph Neural Network (NGNN): https://github.com/muhanzhang/NestedGNN
Attention Free Transformer: https://github.com/rish-16/aft-pytorch
