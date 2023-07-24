import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score,roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle 
torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable

from utils import *
from gnn import *
from gnn_virtual_node import * 
from load_graph_data import DTADataset, collate

class Classifier(nn.Sequential):
    def __init__(self, model_graph, **config):
        super(Classifier, self).__init__()
        self.protein_gnn = model_graph
        
        self.dropout = nn.Dropout(0.1)

        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1

        self.ll = 256 

        #substructure
        dims1 = [self.ll,512,512,1]
        self.predictor1 = nn.ModuleList([nn.Linear(dims1[i], dims1[i+1]) for i in range(3)])

        #embedding, sequence length drug, sequence length protein
        self.proj_p = nn.Linear(1280,128)
        self.proj_d = nn.Linear(512, 128)
        self.proj_p_gnn_unsup = nn.Linear(256, 128)
        self.proj_d_gnn_unsup = nn.Linear(256, 128)

        self.ln = nn.LayerNorm(128)
        self.ln_f = nn.LayerNorm(256)
        self.bn_d = nn.Sequential(
                nn.BatchNorm1d(128),
        )

        self.bn_p = nn.Sequential(
                nn.BatchNorm1d(128),
        )

    def forward(self, input_x0, input_x1):
        bs = input_x0.y.shape[0]
        p_feature = self.proj_p(F.dropout(input_x1.p_feature.reshape(bs,-1),0.1))
        d_feature = self.proj_d(F.dropout(input_x0.d_feature.reshape(bs,-1),0.1))
 
        v_D_graph, v_P_graph = self.protein_gnn(input_x0, input_x1, d_feature, p_feature)

	#prediction
        v_f = torch.cat((v_D_graph, v_P_graph),1).squeeze()
	
        for i, l in enumerate(self.predictor1):
            if i==(len(self.predictor1)-1):
                feat = v_f
                v_f = l(v_f)
            else:
                v_f = l(v_f)
                v_f = F.relu(self.dropout(v_f))
        
        return v_f, feat

def model_initialize(**config):
	model = DTI(**config)
	return model

def model_pretrained(path_dir = None, model = None):
	if model is not None:
		path_dir = download_pretrained_model(model)
	config = load_dict(path_dir)
	model = DTI(**config)
	model.load_pretrained(path_dir + '/model.pt')    
	return model

class DTI:
	'''
		Drug Target Binding Affinity 
	'''

	def __init__(self, **config):
		self.model_graph = EGNN()
		self.model = Classifier(self.model_graph, **config)
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.result_folder = config['result_folder']
		if not os.path.exists(self.result_folder):
			os.mkdir(self.result_folder)            
		self.binary = False

	#prepare the protein and drug pairs
	def test_(self, data_generator, model, repurposing_mode = False, test = False):
		y_pred = []
		y_label = []
		model.eval()
		for i, dataitem in enumerate(data_generator):
			#dataitem.to(device)                        
			label = dataitem[0].y		
			data0 = dataitem[0].to(device)                
			data1 = dataitem[1].to(device)
			score, feat = model(data0, data1)
			has_nan = torch.any(torch.isnan(score))
			if has_nan:
				pdb.set_trace()
			label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)

			if self.binary:
				loss_fct = torch.nn.BCELoss()
				m = torch.nn.Sigmoid()
				if score.shape[0]>1:
					n = torch.squeeze(m(score), 1)
					logits = torch.squeeze(m(score)).detach().cpu().numpy()
				else:
					n = m(score)
					logits = m(score).detach().cpu().numpy()

			label_ids = label.to('cpu').numpy()
			y_label = y_label + label_ids.flatten().tolist()
			y_pred = y_pred + logits.flatten().tolist()
			outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

		model.train()
		if self.binary:
			if repurposing_mode:
				return y_pred
			## ROC-AUC curve
			if test:
				roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
				plt.figure(0)
				draw_roc_curve(y_pred, y_label, roc_auc_file, self.drug_encoding + '_' + self.target_encoding)
				plt.figure(1)
				pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
				prauc_curve(y_pred, y_label, pr_auc_file, self.drug_encoding + '_' + self.target_encoding)
			fpr, tpr, thresholds = roc_curve(y_label, y_pred)

			precision = tpr / (tpr + fpr)
			f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
			#thred_optim = thresholds[5:][np.argmax(f1[5:])]

			y_pred_s = [1 if i else 0 for i in (np.array(y_pred) >= 0.5)]
			#AUROC
			auc_k = auc(fpr, tpr)
			#AUPRC
			auprc = average_precision_score(y_label, y_pred)

			####################################################
			#confusion matrix
			tn, fp, fn, tp = confusion_matrix(y_label, y_pred_s).ravel()
			#recall
			rs =recall_score(y_label, y_pred_s)
			#pre
			pre = precision_score(y_label, y_pred_s)
			total1 = tn+tp+fn+fp
			#####from confusion matrix calculate accuracy
			accuracy1 = (tp+tn) / total1
			sensitivity1 = tp/(tp+fn)
			specificity1 = tn/(tn+fp)

			return auc_k, auprc, sensitivity1, specificity1, accuracy1, f1_score(y_label, outputs), log_loss(y_label,y_pred), y_pred, y_label
		
	def train(self, train, val, test, verbose = True,datanum=0,setnum=0):
		#if len(train.Label.unique()) == 2:
		self.binary = True
		self.config['binary'] = True

		lr = self.config['LR']
		decay = self.config['decay']
		BATCH_SIZE = self.config['batch_size']
		train_epoch = self.config['train_epoch']

		loss_history = []
		val_loss = []
		val_acc = []
	
		self.model = self.model.to(self.device)
		# support multiple GPUs
		if torch.cuda.device_count() > 1:
			if verbose:
				print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
			self.model = nn.DataParallel(self.model, dim = 0)
			self.model = self.model.to(self.device)
		elif torch.cuda.device_count() == 1:
			if verbose:
				print("Let's use " + str(torch.cuda.device_count()) + " GPU!")
		else:
			if verbose:
				print("Let's use CPU/s!")
		# Future TODO: support multiple optimizers with parameters
		

		opt = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = decay)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=0)
		if verbose:
			print('--- Data Preparation ---')

		params = {'batch_size': BATCH_SIZE,
	    		'shuffle': True,
	    		'num_workers': self.config['num_workers'],
	    		'drop_last': False}
		if (self.drug_encoding == "MPNN"):
			params['collate_fn'] = mpnn_collate_func

		train_set = DTADataset(dataFrame=train, setnum=setnum, phase='train')
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                                               collate_fn=collate)
		
		val_set = DTADataset(dataFrame=val, setnum=setnum, phase='valid')
		valid_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                                               collate_fn=collate)
		if test is not None:
			test_set = DTADataset(dataFrame=test, setnum=setnum, phase='test')
			test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

		# early stopping
		if self.binary:
			max_auc = 0
		else:
			max_MSE = 10000

		model_max = copy.deepcopy(self.model)
	
		valid_metric_record = []
		valid_metric_header = ["# epoch"] 
		if self.binary:
			valid_metric_header.extend(["AUROC", "AUPRC", "F1","acc"])
		else:
			valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
		table = PrettyTable(valid_metric_header)
		float2str = lambda x:'%0.4f'%x
		if verbose:
			print('--- Go for Training ---')
		writer = SummaryWriter()
		t_start = time() 
		iteration_loss = 0
		ii = 0
		best_epo = 0
		for epo in range(train_epoch):
			
			for i, dataitem in enumerate(train_loader):
				#dataitem.to(device)
				label = dataitem[0].y
				data0 = dataitem[0].to(device)
				data1 = dataitem[1].to(device)
				
				score, feat = self.model(data0, data1)
				label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)

				if self.binary:
					loss_fct = torch.nn.BCELoss()
					m = torch.nn.Sigmoid()
					n = torch.squeeze(m(score), 1).squeeze()
					loss0 = loss_fct(n, label)

					loss = loss0
				loss_history.append(loss.item())
				writer.add_scalar("Loss/train", loss.item(), iteration_loss)
				iteration_loss += 1

				opt.zero_grad()
				loss.backward()
				opt.step()

				if verbose:
					if (i % 100 == 0):
						t_now = time()
						print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
							' with loss ' + str(loss.cpu().detach().numpy())[:7] +\
							". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 
						### record total run time
						

			##### validate, select the best model up to now 
			with torch.set_grad_enabled(False):
				
				if self.binary:  
					## binary: ROC-AUC, PR-AUC, F1, cross-entropy loss
					auc, auprc, sensitivity, specificity, acc, f1, loss, logits, _ = self.test_(valid_loader, self.model)	
					lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, f1, acc]))
					valid_metric_record.append(lst)
				
					if auc > max_auc:
						ii = epo
						model_max = copy.deepcopy(self.model)
						max_auc = auc
					best_epo = epo

					if verbose:
						print('Validation at Epoch '+ str(epo + 1) + ', AUROC: ' + str(auc)[:7] + \
						  ' , AUPRC: ' + str(auprc)[:7] +' , sensitivity: ' +str(sensitivity)+' , specificity: ' +str(specificity)+ ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
						  str(loss)[:7])
					val_loss.append(loss)
					val_acc.append(acc)
				
					table.add_row(lst)
				
					if epo%20==0:
		
						if test is not None:
							if verbose:
								print('--- Go for Testing ---')
							if self.binary:
								auc, auprc, sensitivity,specificity,acc,f1,loss,logits,_ = self.test_(test_loader, model_max, test=True)
								test_table = PrettyTable(["AUROC", "AUPRC", "F1","Sensitivity","Specificity"])
								test_table.add_row(list(map(float2str, [auc, auprc, f1,sensitivity,specificity])))
								if verbose:
									print('Test at Epoch '+ str(epo + 1) + ', AUROC: ' + str(auc)[:7] + \
									' , AUPRC: ' + str(auprc)[:7] +' , sensitivity: ' +str(sensitivity)+' , specificity: ' +str(specificity)+ ',acc:'+str(acc)+' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
									str(loss)[:7])				   
				
						######### learning record ###########

						### 1. test results
						prettytable_file = os.path.join(self.result_folder, "test_markdowntable"+str(epo)+".txt")
						with open(prettytable_file, 'w') as fp:
							fp.write(test_table.get_string())

		#### after training 
		prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
		with open(prettytable_file, 'w') as fp:
			fp.write(table.get_string())

		# load early stopped model
		self.model = model_max
		print('model epoch:'+str(ii))

		with torch.set_grad_enabled(False):
			if verbose:
				print('--- Go for Testing ---')
			if self.binary:
				auc, auprc, sensitivity,specificity,acc,f1,loss,test_pred, test_label = self.test_(test_loader, model_max, test = True)
				test_table = PrettyTable(["AUROC", "AUPRC", "F1","Sensitivity","Specificity"])
				test_table.add_row(list(map(float2str, [auc, auprc, f1,sensitivity,specificity])))
				test_table.add_row(['1','1','1','1',str(best_epo)])
				if verbose:
					print('Test at Epoch: '+ str(epo + 1) + ', AUROC: ' + str(auc)[:7] + \
						' , AUPRC: ' + str(auprc)[:7] +' , sensitivity: ' +str(sensitivity)+' , specificity: ' +str(specificity)+ ',acc:'+str(acc)+ ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
						str(loss)[:7])	

			
			np.save(os.path.join(self.result_folder, str(self.drug_encoding) + '_' + str(self.target_encoding) 
				     + '_logits.npy'), np.array(logits))
			out={'res':logits}
			Out = pd.DataFrame(out)
			Out.to_excel(os.path.join(self.result_folder, str(self.drug_encoding) + '_' + str(self.target_encoding) 
				     + '_logits.xls'))
			test = pd.DataFrame({'label':test_label, 'pred':test_pred})
			test.to_csv(self.result_folder +'/'+str(datanum)+'_'+str(setnum)+'.csv')
			######### learning record ###########

			### 1. test results
			prettytable_file = os.path.join(self.result_folder, "test_markdowntable_"+str(datanum)+'_'+str(setnum)+'_'+str(self.config['LR'])+'_'+str(self.config['batch_size'])+'_'+str(self.config['hidden_dim_drug'])+'_'+str(self.config['hidden_dim_protein'])+'_'+str(self.config['kernel_dim'])+".txt")
			with open(prettytable_file, 'w') as fp:
				fp.write(test_table.get_string())

		### 2. learning curve 
		fontsize = 16
		loss_history = loss_history[300:]
		iter_num = list(range(1,len(loss_history)+1))
		plt.figure(3)
		plt.plot(iter_num, loss_history, "bo-")
		plt.xlabel("iteration", fontsize = fontsize)
		plt.ylabel("loss value", fontsize = fontsize)
		pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
		with open(pkl_file, 'wb') as pck:
			pickle.dump(loss_history, pck)

		fig_file = os.path.join(self.result_folder, "loss_curve.png")
		print(fig_file)
		plt.savefig(fig_file)

		iter_num = list(range(1,len(val_loss)+1))
		plt.figure(4)
		plt.plot(iter_num, val_loss, "bo-")
		plt.xlabel("iteration", fontsize = fontsize)
		plt.ylabel("loss value", fontsize = fontsize)
		fig_file = os.path.join(self.result_folder, "val_loss_curve.png")
		print(fig_file)
		plt.savefig(fig_file)

		iter_num = list(range(1,len(val_acc)+1))
		plt.figure(5)
		plt.plot(iter_num, val_acc, "bo-")
		plt.xlabel("iteration", fontsize = fontsize)
		plt.ylabel("loss value", fontsize = fontsize)
		fig_file = os.path.join(self.result_folder, "val_acc_curve.png")
		print(fig_file)
		plt.savefig(fig_file)

		if verbose:
			print('--- Training Finished ---')
			writer.flush()
			writer.close()
		model_file = os.path.join(self.result_folder,'DAVIS_'+str(setnum)+'.pth')
		torch.save(model_max.state_dict(), model_file) 

	def predict(self, df_data):
		print('predict') 
		
	def save_model(self, path_dir):
		if not os.path.exists(path_dir):
			os.makedirs(path_dir)
		torch.save(self.model.state_dict(), path_dir + '/model.pt')
		save_dict(path_dir, self.config)

	def load_pretrained(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

		if self.device == 'cuda':
			state_dict = torch.load(path)
		else:
			state_dict = torch.load(path, map_location = torch.device('cpu'))
		# to support training from multi-gpus data-parallel:
        
		if next(iter(state_dict))[:7] == 'module.':
			# the pretrained model is from data-parallel module
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				name = k[7:] # remove `module.`
				new_state_dict[name] = v
			state_dict = new_state_dict

		self.model.load_state_dict(state_dict)

		self.binary = self.config['binary']


