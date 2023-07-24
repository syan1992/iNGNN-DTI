# Prediction of Drug - Target Interaction with Interpretable Nested Graph Neural Network and Pretrained Molecule Models (iNGNN-DTI)
##

## Install

To install the required packages, follow these instructions (tested on a linux terminal):

1- clone the repository

```
git clone https://github.com/syan1992/iNGNN-DTI.git
```

2- cd into the cloned directory

```
cd iNGNN-DTI
```

3- run the install script
```
pip install -r requirements.txt
```

We run the code on GPU.

## Usage example
We list all command lines in the shell script 'autorun.sh' for the seven datasets (freesolv, delaney, lipophilicity, bace, sider, tox21, clintox) we test in our experiments. 
Run 'autorun.sh' with the name of the dataset as a parameter.
```sh
./autorun.sh freesolv
```
We save the model with the best performance on the validation set and evaluate the best model with the test set.
Both model and test results will be saved in the 'save' folder.

## Hyperparameters
Some specific hyperparameters in this work,  
|  Name   | Description  |
| :---        |    :----:   |
|  wscl  | The weight of the supervised contrastive loss in the loss function. Suggest to test values in [0.1 to 1]|
| wrecon  | The weight of the reconstruction loss in the loss function. Suggest to test values in [0.1 to 1]|
| gamma1  | The hyperparameter of the weighted supervised contrastive loss for the regression task. Suggest to test values in [2,3,4] |
| gamma2  | The hyperparameter of the weighted supervised contrastive loss for the regression task. Suggest to test values in [1,2,3] |

## Acknowledgement
Supervised contrastive learning : https://github.com/HobbitLong/SupContrast  
Deepgcn : https://github.com/lightaime/deep_gcns_torch

## Reference
```
@inproceedings{sun2022molecular,
  title={Molecular Property Prediction based on Bimodal Supervised Contrastive Learning},
  author={Sun, Yan and Islam, Mohaiminul and Zahedi, Ehsan and Kuenemann, M{\'e}laine and Chouaib, Hassan and Hu, Pingzhao},
  booktitle={2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={394--397},
  year={2022},
  organization={IEEE}
}
```
