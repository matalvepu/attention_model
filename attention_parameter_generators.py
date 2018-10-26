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

lan_hidden_list=[128,256]
audio_hidden_list=[40,48]
face_hidden_list=[36,30]
learnig_rate_list=[0.00001,0.000055,0.0001]
drop_out_list=[0.2]
params_list=[]

for i in itertools.product(lan_hidden_list,audio_hidden_list,face_hidden_list,learnig_rate_list,drop_out_list):
	params_list.append(i)

# print "len parm",len(params_list)

len_params=range(len(params_list))
shuffle(len_params)
new_params_list=[]
for index in len_params:
	new_params_list.append(params_list[index])


# # print params_list[0:10]
# print new_params_list[0:10]
# print len(new_params_list)


param_1=new_params_list

# write_pkl(new_params_list,"params_set.pkl")


lan_hidden_list=[128,256]
audio_hidden_list=[40,48]
face_hidden_list=[36,30]
learnig_rate_list=[0.00001,0.000055,0.0001]
drop_out_list=[0.15]
params_list=[]

for i in itertools.product(lan_hidden_list,audio_hidden_list,face_hidden_list,learnig_rate_list,drop_out_list):
	params_list.append(i)

# print "len parm",len(params_list)

len_params=range(len(params_list))
shuffle(len_params)
new_params_list=[]
for index in len_params:
	new_params_list.append(params_list[index])


# # print params_list[0:10]
# print new_params_list[0:10]
# print len(new_params_list)

param_2=new_params_list

final_param_list=param_1+param_2


print len(final_param_list)

print "param_1",final_param_list[0:10]

print "param_2",final_param_list[-10:]

# write_pkl(final_param_list,"params_set.pkl")