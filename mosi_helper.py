import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence

import time
import numpy as np

helper_gpu_mode=True

w_dim_index=range(0,300)
#w_dim_index=[0,1,2]
opensmile_dim_index=range(300,1885)

# facet_dim_index=range(1885,1932)
facet_dim_index=range(374,421)
# facet_dim_index=[3,4]
f_dim_index=w_dim_index+facet_dim_index

def variablize(tensor_input):
    '''
    Turn numpy to variable. put to gpu if necessary
    '''
    if (helper_gpu_mode and torch.cuda.is_available()):
        return Variable(tensor_input.cuda())
    else:
        return Variable(tensor_input)

def filter_train_features(x):
    x=np.array(x)
    # x=np.take(x,f_dim_index)
    x_lan=np.take(x,w_dim_index)
    x_face=np.take(x,facet_dim_index)
    return x_lan,x_face