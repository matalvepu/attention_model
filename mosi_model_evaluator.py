import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence

import time
import numpy as np
from attention_model import LSTM_custom,MOSI_attention_classifier
import gzip, cPickle
import matplotlib.pyplot as plt
from mosi_helper import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class MosiEvaluator():
	
	def evaluate(self,model,feature_x,target_y):
		target_y=[y[0] for y in target_y]
		model.eval()
		predicted_y=[]

		with torch.no_grad():
			for i in range(len(feature_x)):
				x=feature_x[i]
				y_hat=model.forward(x)
				y_hat = torch.sigmoid(y_hat[0][0])

				if y_hat>=0.5:
					predicted_y.append(1)
				else:
					predicted_y.append(0)

		acc =  accuracy_score(target_y,predicted_y)
		f1 = f1_score(target_y,predicted_y)
		precision  = precision_score(target_y,predicted_y)
		recall = recall_score(target_y,predicted_y)

		return [acc,precision,recall,f1]

