import os
os.chdir('../')
os.environ['CUDA_VISIBLE_DEVICES']="1"
import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
import numpy as np
import pdb
X_drug, X_target, y = load_process_DAVIS('./data/', binary=False)

drug_encoding = 'CNN1D'#'CNN'
target_encoding = 'CNN1D'#'Transformer'
#train, val, test = data_process(X_drug, X_target, y,
#                                drug_encoding, target_encoding,
#                                split_method='random',frac=[0.7,0.1,0.2])

# use the parameters setting provided in the paper: https://arxiv.org/abs/1801.10193
config = generate_config(drug_encoding = drug_encoding,
                         target_encoding = target_encoding,
                         cls_hidden_dims = [1024,1024,512],
                         train_epoch = 50,
                         LR = 0.0001,
                         batch_size = 64,
                         cnn_drug_filters = [32,64,128],
                         cnn_target_filters = [32,64,128],
                         cnn_drug_kernels = [4,4,4],
                         cnn_target_kernels = [8,8,8],
                         result_folder = '/home/yan/code/DTI/DTI_test_module_new_1/DeepPurpose/result'
                        )


model = models.model_initialize(**config)

def get_task(task_name):
    if task_name.lower() == 'biosnap':
        return 'dataset/BIOSNAP/full_data'
    elif task_name.lower() == 'kiba':
        return 'dataset/KIBA'
    elif task_name.lower() == 'davis':
        return 'dataset/DAVIS'

dataFolder = ['/home/yan/code/DTI/datasetDTI/DAVIS/unseen_prot/','/home/yan/code/DTI/datasetDTI/BIOSNAP/5 fold/','/home/yan/code/DTI/datasetDTI/KIBA/5 fold/']

param_list=[
{'lr':0.0001,'batch_size':16,'hidden_dim':256,'kernel_dim':3},
]
'''
protein_feature = np.load('/home/yan/code/DTI/CNN_Transformer_DTI_1/protein_feature.npy', allow_pickle=True).item()
def Get_Protein_Feature(data):
    p_list = data['Target Sequence']
    feature=[]
    for p in p_list:
        p = str(p[0:1022])
        feature.append(protein_feature[p])
    print(len(feature))
    return feature
'''

for j in [0]:
    for i in [0]:
        for k in param_list:
            train = pd.read_csv(dataFolder[j] + '/train'+str(i)+'.csv')
            val = pd.read_csv(dataFolder[j] + '/val'+str(i)+'.csv')
            test = pd.read_csv(dataFolder[j] + '/test'+str(i)+'.csv')
            config['LR'] = k['lr']
            config['batch_size'] = k['batch_size']
            config['hidden_dim_drug'] = k['hidden_dim']
            config['hidden_dim_protein'] = k['hidden_dim']
            config['kernel_dim'] = k['kernel_dim']
            model = models.model_initialize(**config)
            model.train(train, val, test, datanum=j,setnum=i)
