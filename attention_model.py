import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
import time
import numpy as np
from mosi_helper import *


class ModelIO():
    '''
    The ModelIO class implements a load() and a save() method that
    makes model loading and saving easier. Using these functions not
    only saves the state_dict but other important parameters as well from
    __dict__. If you instantiate from this class, please make sure all the
    required arguments of the __init__ method are actually saved in the class
    (i.e. self.<param> = param). 
    That way, it is possible to load a model with the default parameters and
    then change the parameters to correct values from stored in the disk.
    '''
    ignore_keys = ['_backward_hooks','_forward_pre_hooks','_backend',\
        '_forward_hooks']#,'_modules','_parameters','_buffers']
    def save(self, fout):
        '''
        Save the model parameters (both from __dict__ and state_dict())
        @param fout: It is a file like object for writing the model contents.
        '''
        model_content={}
        # Save internal parameters
        for akey in self.__dict__:
            if not akey in self.ignore_keys:
                model_content.update(self.__dict__)
        # Save state dictionary
        model_content['state_dict']=self.state_dict()
        torch.save(model_content,fout)

    def load(self,fin,map_location=None):
        '''
        Loads the parameters saved using the save method
        @param fin: It is a file-like obkect for reading the model contents.
        @param map_location: map_location parameter from
        https://pytorch.org/docs/stable/torch.html#torch.load
        Note: although map_location can move a model to cpu or gpu,
        it doesn't change the internal model flag refering gpu or cpu.
        '''
        data=torch.load(fin,map_location)
        self.__dict__.update({key:val for key,val in data.items() \
            if not key=='state_dict'})
        self.load_state_dict(data['state_dict'])


class LSTM_custom(nn.Module):
    '''
    A custom implementation of LSTM in pytorch. Donot use. VERY slow
    '''
    def __init__(self,input_dim,hidden_dim,context_dim):

        super(LSTM_custom,self).__init__()
        self.W_xi = nn.Linear(input_dim,hidden_dim)
        self.W_hi = nn.Linear(hidden_dim,hidden_dim)
        self.W_xf = nn.Linear(input_dim,hidden_dim)
        self.W_hf = nn.Linear(hidden_dim,hidden_dim)
        self.W_zf = nn.Linear(context_dim,hidden_dim)
        self.W_xg = nn.Linear(input_dim,hidden_dim)
        self.W_hg = nn.Linear(hidden_dim,hidden_dim)
        self.W_zg = nn.Linear(context_dim,hidden_dim)
        self.W_xo = nn.Linear(input_dim,hidden_dim)
        self.W_ho = nn.Linear(hidden_dim,hidden_dim)
        self.W_zo = nn.Linear(context_dim,hidden_dim)
        self.W_zi = nn.Linear(context_dim,hidden_dim)
        self.drop = nn.Dropout(0.1)


    def dropout(self):
        s_dict=self.W_hi.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_hi.load_state_dict(s_dict)

        s_dict=self.W_hf.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_hf.load_state_dict(s_dict)

        s_dict=self.W_hg.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_hg.load_state_dict(s_dict)

        s_dict=self.W_ho.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_ho.load_state_dict(s_dict)


    def forward(self,x,hidden,z):
        h,c = hidden[0],hidden[1]
        i = torch.sigmoid(self.W_xi(x) + self.W_hi(h) + self.W_zi(z))
        f = torch.sigmoid(self.W_xf(x) + self.W_hf(h) + self.W_zf(z))
        g = torch.tanh(self.W_xg(x) + self.W_hg(h) + self.W_zg(z))
        o = torch.sigmoid(self.W_xo(x)+self.W_ho(h)+ self.W_zo(z))
        c_ = f*c + i*g
        h_ = o * torch.tanh(c_)
        return h_,c_




class MOSI_attention_classifier(nn.Module,ModelIO):

    def init_lstm_param(self,hidden_dim):
        return (variablize(torch.zeros(1, hidden_dim)),variablize(torch.zeros(1, hidden_dim)))

    def init_context_var(self,context_dim):
        return variablize(torch.zeros(1, context_dim))

    def __init__(self,lan_param,face_param,out_dim):
        super(MOSI_attention_classifier,self).__init__()
        self.lan_lstm=LSTM_custom(lan_param['input_dim'],lan_param['hidden_dim'],lan_param['context_dim'])
        self.face_lstm=LSTM_custom(face_param['input_dim'],face_param['hidden_dim'],face_param['context_dim'])
        self.context_dim = lan_param['context_dim']
        self.hidden_comb_dim=lan_param['hidden_dim']+face_param['hidden_dim']
        self.W_a = nn.Linear(self.context_dim,self.hidden_comb_dim)
        self.lan_init_param = self.init_lstm_param(lan_param['hidden_dim'])
        self.face_init_param = self.init_lstm_param(face_param['hidden_dim'])      
        self.W_ac = nn.Linear(self.hidden_comb_dim,self.context_dim)
        self.W_cout = nn.Linear(self.context_dim,out_dim)
        self.z_init = self.init_context_var(self.context_dim)
        # print "z_init",self.z_init

    def forward(self,opinion):
        s=nn.Softmax()
        for i,x in enumerate(opinion):
            x_lan,x_face=filter_train_features(x)
            x_lan=variablize(torch.FloatTensor(x_lan))
            x_face=variablize(torch.FloatTensor(x_face))

            #lhtmstep 
            if i==0:
                if self.training:
                    self.lan_lstm.dropout()
                    self.face_lstm.dropout()
                h_lan,c_lan=self.lan_lstm.forward(x_lan,self.lan_init_param,self.z_init)
                h_face,c_face=self.face_lstm.forward(x_face,self.face_init_param,self.z_init)
                h_i=torch.cat((h_lan,h_face),1)
                attn_weigths=s(self.W_a(self.z_init))
                h_atten = torch.mul(h_i,attn_weigths)
                z=self.W_ac(h_atten)
            else:
                h_lan,c_lan=self.lan_lstm.forward(x_lan,(h_lan,c_lan),z)
                h_face,c_face=self.face_lstm.forward(x_face,(h_face,c_face),z)
                h_i=torch.cat((h_lan,h_face),1)
                attn_weigths=s(self.W_a(z))
                h_atten = torch.mul(h_i,attn_weigths)
                z=self.W_ac(h_atten)

            # print "####DEBUG#####", i
            # print "h_lan",h_lan
            # print "h_face",h_face
            # print "h_i",h_i
            # print "atten_weights",attn_weigths
            # print "h_atten",h_atten
            # print "z",z 

        out = self.W_cout(z)
        # print "out", out
        return out


def test_model():
    opinion=np.array([[1,2,3,4,5],[3,4,5,6,7]])
    lan_param={'input_dim':3,'hidden_dim':2,'context_dim':2}
    face_param={'input_dim':2,'hidden_dim':2,'context_dim':2}
    out_dim=1

    mosi_model=MOSI_attention_classifier(lan_param,face_param,out_dim)
    # print mosi_model
    mosi_model.forward(opinion)



# test_model()

# class LIWC_lie_detector(nn.Module,ModelIO):

#     def init_hidden(self,hidden_dim):
#         return (variablize(torch.zeros(1, hidden_dim)),variablize(torch.zeros(1, hidden_dim)))

#     def __init__(self,input_dim,lstm_hidden_dim,out_dim):
#         super(LIWC_lie_detector,self).__init__()
#         self.lstm=LSTM_custom(input_dim,lstm_hidden_dim)
#         self.W_fhout = nn.Linear(lstm_hidden_dim,out_dim)
#         self.h = self.init_hidden(lstm_hidden_dim)

#     def forward(self,opinion):
#         for i,x in enumerate(opinion):
#             x=variablize(torch.FloatTensor(filter_train_features(x)))
#             if i==0:
#                 if self.training:
#                     self.lstm.dropout()
#                 h_x,c_x=self.lstm.forward(x,self.h)
#             else:
#                 h_x,c_x=self.lstm.forward(x,(h_x,c_x))

#         out=self.W_fhout(h_x)
#         # return torch.sigmoid(out)
#         return out
