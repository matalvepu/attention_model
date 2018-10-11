import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import gzip, cPickle

class MosiDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        Y = torch.FloatTensor(self.Y[idx])
        return X,Y


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = map(lambda (x, y):
                    (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], batch))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


def load_data(file_name):
	fp=gzip.open(file_name,'rb') 
	(x,y) =cPickle.load(fp)
	fp.close()
	return x,y

def get_data_loader(x,y):
	data=MosiDataset(x,y)
	d_loader=DataLoader(data, batch_size=20, shuffle=True, num_workers=10, collate_fn=PadCollate(dim=0))
	return d_loader


def get_unpad_data(x):
	x=x.numpy()
	x=x[~np.all(x == 0, axis=1)]
	return x


# valid_x,valid_y=load_data('../mosi_data/COVAREP/valid_matrix.pkl')
# # print valid_x 
# # print valid_y
# x=np.array(valid_x[0])
# print x.shape
# print x
# d_loader=get_data_loader(valid_x,valid_y)

# for i, data in enumerate(d_loader):
# 	seq , label = data
# 	# print seq
# 	# print label
# 	for j,x in enumerate(seq):
# 		print x 
# 		print torch.FloatTensor([label[j]])


# 	# print get_unpad_data(seq[0])
# 	# print get_unpad_data(seq[1])

# 	break


# len train 1283
# loaded test
# len test 686
# loaded valid
# len valid 229

# train_x,train_y=load_data('../mosi_data/COVAREP/valid_matrix.pkl')
# train_data_loader=get_data_loader(train_x,train_y)

# print("loaded train data loader")
# test_x,test_y=train_x[0:30],train_y[0:30]
# print len(test_x)
# valid_x,valid_y=train_x[30:50],train_y[30:50]
# print("loaded valid")
# print len(valid_x)
# train_x,train_y=train_x[50:],train_y[50:]
# print len(train_x)