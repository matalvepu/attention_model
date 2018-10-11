import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import numpy as np
from attention_model import LSTM_custom,MOSI_attention_classifier
import gzip, cPickle
import matplotlib.pyplot as plt
from mosi_helper import *
from mosi_model_evaluator import MosiEvaluator
import datetime
import csv 

model_version="~/ets/deep_learning/experiment/attention_model/data_loader/"
# time_stamp=str(datetime.datetime.now())

mini_batch_size=10	

def load_data(file_name):
	fp=gzip.open(file_name,'rb') 
	data =cPickle.load(fp)
	fp.close()
	return data["x"],data["y"]

def save_result(model_name,eval_results,params):
	print params

	eval_results=[model_name]+eval_results

	for key,value in params.iteritems():
		print value
		eval_results.append(value)

	result_csv_file = model_version+"results/all_results.csv"	
	with open(result_csv_file, 'a') as out_f:
		wr = csv.writer(out_f)
		wr.writerow(eval_results)
	out_f.close()

def print_loss(e_tr_losses,e_val_losses,model_name):
	fig_name=model_version+"fig/"+model_name+".png"
	legend=["train_loss","val_loss"]
	plt.plot(e_tr_losses)
	plt.plot(e_val_losses)
	plt.ylabel('Loss')
	plt.xlabel('iteration')
	plt.legend(legend, loc='upper right')
	title="Loss plot for "+model_name
	plt.title(title)
	plt.savefig(fig_name)
	plt.close()

train_x,train_y=load_data('../mosi_data/COVAREP/train_matrix.pkl')
print("loaded train")
test_x,test_y=load_data('../mosi_data/COVAREP/test_matrix.pkl')
print("loaded test")
valid_x,valid_y=load_data('../mosi_data/COVAREP/valid_matrix.pkl')
print("loaded valid")

# train_x=valid_x[:10]
# train_y=valid_y[:10]

# valid_x=valid_x[10:13]
# valid_y=valid_y[10:13]

# test_x=valid_x[13:16]
# test_y=valid_y[13:16]

def get_mini_batch_list(batch_size):
	index_arr=np.arange(len(train_x))

	np.random.shuffle(index_arr)
	index_arr=index_arr.tolist()

	mini_batch_list=[]
	for i in range(0,len(index_arr),batch_size):
		mini_batch_list.append(index_arr[i:i+batch_size])

	return mini_batch_list


def train_epoch_minibatch(mosi_model,opt,criterion):
	losses = []

	mini_batch_list=get_mini_batch_list(mini_batch_size)
	mosi_model.train()
	for mini_batch in mini_batch_list:		
		opt.zero_grad()
		mini_batch_losses=[]
		for i in mini_batch:
			x=train_x[i]
			y=variablize(torch.FloatTensor([train_y[i]]))
			y_hat=mosi_model.forward(x)
			loss = criterion(y_hat, y)
			mini_batch_losses.append(loss)

		mini_batch_loss=reduce(torch.add,mini_batch_losses)/len(mini_batch_losses)
		mini_batch_loss.backward()
		opt.step()
		losses.append(mini_batch_loss.cpu().data.numpy())

	return np.nanmean(losses)


def train_epoch(mosi_model,opt,criterion):
	losses = []
	mosi_model.train()
	for i in range(len(train_x)):
		x=train_x[i]
		y=variablize(torch.FloatTensor(train_y[i]))
		opt.zero_grad()

		y_hat=mosi_model.forward(x)
		loss = criterion(y_hat, y)

		loss.backward()
		opt.step()
		losses.append(loss.cpu().data.numpy())

	return np.nanmean(losses)


def validation_loss(mosi_model,criterion):

	mosi_model.eval()
	losses = []
	with torch.no_grad():
		for i in range(len(valid_x)):
			x=valid_x[i]
			y=variablize(torch.FloatTensor([valid_y[i]]))
			y_hat=mosi_model.forward(x)
			loss = criterion(y_hat, y)
			losses.append(loss.cpu().data.numpy())

	return np.nanmean(losses)


def evaluate_best_model(model_name,params):

	evaluator=MosiEvaluator()
	model_file=model_version+"models/"+model_name

	d_lan_param={'input_dim':1,'hidden_dim':1,'context_dim':1}
	d_face_param={'input_dim':1,'hidden_dim':1,'context_dim':1}
	
	best_model=MOSI_attention_classifier(d_lan_param,d_face_param,1)
	best_model.load(open(model_file,'rb'))

	comment="validtion evaluation for best model: "+model_name
	print(comment)
	eval_val = evaluator.evaluate(best_model,valid_x,valid_y)
	comment="test evaluation for best model: "+model_name
	print(comment)
	eval_test = evaluator.evaluate(best_model,test_x,test_y)

	eval_results=eval_val+eval_test
	save_result(model_name,eval_results,params)



def train_mosi_sentiments(mosi_model,params):

	evaluator=MosiEvaluator()

	model_name="params_"+str(params)
	model_file=model_version+"models/"+model_name

	opt = optim.Adam(mosi_model.parameters(), lr=params['learn_rate'])
	criterion = nn.BCEWithLogitsLoss()
	e_tr_losses = []
	e_val_losses = []
	num_epochs = 40

	best_valid_loss=np.inf

	for e in range(num_epochs):
		train_loss=train_epoch_minibatch(mosi_model, opt, criterion)
		e_tr_losses.append(train_loss)

		valid_loss=validation_loss(mosi_model,criterion)
		e_val_losses.append(valid_loss)

		if valid_loss<best_valid_loss:
			best_valid_loss=valid_loss			
			mosi_model.save(open(model_file,'wb'))	

		if (e%5==0):
			print_loss(e_tr_losses,e_val_losses,model_name)

	evaluate_best_model(model_name,params)

if __name__=='__main__':
	print("started")
	start_time = time.time()

	out_dim=1

	# lan_hid_dim=200
	# face_hid_dim=32
	# context_dim=128
	# learning_rate=0.0001

	# lan_hid_dim_list=[250,200,172]
	# face_hid_dim_list=[32,24,16]
	# context_dim_list=[128,170,256]
	# learnig_rate_list=[0.00066,0.0066,0.0033,0.0001,0.001,0.01]
	lan_hid_dim_list=[250]
	face_hid_dim_list=[32]
	context_dim_list=[128]
	learnig_rate_list=[0.00066]
	
	for lan_hid_dim in lan_hid_dim_list:
		for face_hid_dim in face_hid_dim_list:
			for context_dim in context_dim_list:
				for learning_rate in learnig_rate_list:
					lan_param={'input_dim':len(w_dim_index),'hidden_dim':lan_hid_dim,'context_dim':context_dim}
					face_param={'input_dim':len(facet_dim_index),'hidden_dim':face_hid_dim,'context_dim':context_dim}

					params={'lan_hid_dim':lan_hid_dim,'face_hid_dim':face_hid_dim,'contxt_dim':context_dim,'learn_rate':learning_rate}

					if (helper_gpu_mode and torch.cuda.is_available()):
						print("gpu found")
						mosi_model=MOSI_attention_classifier(lan_param,face_param,out_dim).cuda()
					else:
						mosi_model=MOSI_attention_classifier(lan_param,face_param,out_dim)

					train_mosi_sentiments(mosi_model,params)

	# dim_params={'lan_hid_dim':lan_hid_dim,'face_hid_dim':face_hid_dim,'context_dim':context_dim}

	# lan_param={'input_dim':len(w_dim_index),'hidden_dim':lan_hid_dim,'context_dim':context_dim}
	# face_param={'input_dim':len(facet_dim_index),'hidden_dim':face_hid_dim,'context_dim':context_dim}

	# params={'lan_hid_dim':lan_hid_dim,'face_hid_dim':face_hid_dim,'contxt_dim':context_dim,'learn_rate':learning_rate}

	# if (helper_gpu_mode and torch.cuda.is_available()):
	# 	print("gpu found")
	# 	mosi_model=MOSI_attention_classifier(lan_param,face_param,out_dim).cuda()
	# else:
	# 	mosi_model=MOSI_attention_classifier(lan_param,face_param,out_dim)

	# train_mosi_sentiments(mosi_model,params)

	# learnig_rate_list=[0.00066,0.0066,0.0033,0.0001,0.001,0.01]
	# hidden_dim_list=[200,128,100,64]	
	
	# for learnig_rate in learnig_rate_list:
	# 	for hidden_dim in hidden_dim_list:

	# 		if (helper_gpu_mode and torch.cuda.is_available()):
	# 			print("gpu found")
	# 			mosi_model=MOSI_sentiment_predictor(input_dim,hidden_dim,out_dim).cuda()
	# 		else:
	# 			mosi_model=MOSI_sentiment_predictor(input_dim,hidden_dim,out_dim)

	# 		train_mosi_sentiments(mosi_model,learnig_rate,hidden_dim)
	time_str="data loader---program run time "+str((time.time() - start_time))+"seconds ---"
	f_name=model_version+"/out.txt"
	f=open(f_name,"a")
	f.write(time_str)
			
	#print("---program run time  %s seconds ---" % (time.time() - start_time))
