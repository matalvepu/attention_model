import itertools
from random import shuffle
import cPickle as pkl 
import gzip


def write_pkl(features,file_name):
	fp=gzip.open(file_name,'wb')
	pkl.dump(features,fp)
	fp.close()


lan_param={'input_dim':3,'hidden_dim':2,'context_dim':2}
audio_param={'input_dim':3,'hidden_dim':2,'context_dim':2}
face_param={'input_dim':2,'hidden_dim':2,'context_dim':2}
context_dim=2
num_atten=2
out_dim=1

lan_hidden_list=[60,100,128,156,184,228,256]
audio_hidden_list=[16,32,48,60]
face_hidden_list=[16,25,32,40]
learnig_rate_list=[0.000066,0.000363,0.00066,0.00363,0.0066,0.00495,0.0001]
params_list=[]
for i in itertools.product(lan_hidden_list,audio_hidden_list,face_hidden_list,learnig_rate_list):
	params_list.append(i)


len_params=range(len(params_list))
shuffle(len_params)


new_params_list=[]
for index in len_params:
	new_params_list.append(params_list[index])


# # print params_list[0:10]
print new_params_list[0:10]
print len(new_params_list)

# write_pkl(new_params_list,"params_set.pkl")
