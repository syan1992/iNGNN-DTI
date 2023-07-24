import os
os.environ['CUDA_VISIBLE_DEVICES']="1"

import numpy as np

import DTI as models
from utils import *
from dataset import *

X_drug, X_target, y = load_process_DAVIS('./data/', binary=False)

drug_encoding = 'CNN1D'
target_encoding = 'CNN1D'

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
                         result_folder = 'result'
                        )


model = models.model_initialize(**config)

def get_task(task_name):
    if task_name.lower() == 'biosnap':
        return 'dataset/BIOSNAP/full_data'
    elif task_name.lower() == 'kiba':
        return 'dataset/KIBA'
    elif task_name.lower() == 'davis':
        return 'dataset/DAVIS'

dataFolder = ['datasetDTI/DAVIS/5 fold/','datasetDTI/BIOSNAP/5 fold/','datasetDTI/KIBA/5 fold/']

param_list=[
{'lr':0.0001,'batch_size':16,'hidden_dim':256,'kernel_dim':3},
]

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
