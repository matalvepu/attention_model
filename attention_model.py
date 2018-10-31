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
        try:
            torch.save(model_content,fout)
        except:
            time.sleep(5)
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
    def __init__(self,input_dim,hidden_dim,context_dim,drop_out):

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
        self.error_check=True
        self.drop = nn.Dropout(drop_out)

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

        s_dict=self.W_zi.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_zi.load_state_dict(s_dict)

        s_dict=self.W_zf.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_zf.load_state_dict(s_dict)

        s_dict=self.W_zg.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_zg.load_state_dict(s_dict)

        s_dict=self.W_zo.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.W_zo.load_state_dict(s_dict)


    def forward(self,x,hidden,z):
        h,c = hidden[0],hidden[1]
        i = torch.sigmoid(self.W_xi(x) + self.W_hi(h) + self.W_zi(z))
        f = torch.sigmoid(self.W_xf(x) + self.W_hf(h) + self.W_zf(z))
        g = torch.tanh(self.W_xg(x) + self.W_hg(h) + self.W_zg(z))
        o = torch.sigmoid(self.W_xo(x)+self.W_ho(h)+ self.W_zo(z))
        c_ = f*c + i*g
        h_ = o * torch.tanh(c_)

        if torch.isnan(h_[0][0]) and self.error_check:
            self.error_check=False
            print "***********h nan found****************"

        return h_,c_


class Memory_attention_network(nn.Module):

    def __init__(self,hidden_comb_dim,context_dim,lan_hid_dim,audio_hid_dim,face_hid_dim,num_atten,drop_out):

        super(Memory_attention_network,self).__init__()
        self.hidden_comb_dim=hidden_comb_dim
        self.num_atten=num_atten

        self.att_w1=nn.Linear(hidden_comb_dim,hidden_comb_dim)
    	self.att_w2=nn.Linear(hidden_comb_dim,hidden_comb_dim)
    	self.att_w3=nn.Linear(hidden_comb_dim,hidden_comb_dim)
        self.att_w4=nn.Linear(hidden_comb_dim,hidden_comb_dim)

        self.l_d=lan_hid_dim
        self.a_d=audio_hid_dim
        self.f_d=face_hid_dim
    	
        self.C_l=nn.Linear(num_atten*lan_hid_dim,lan_hid_dim)
        self.C_a=nn.Linear(num_atten*audio_hid_dim,audio_hid_dim)
        self.C_f=nn.Linear(num_atten*face_hid_dim,face_hid_dim)

        self.W_ac = nn.Linear(hidden_comb_dim,context_dim)

        self.s=nn.Softmax(dim=1)
        self.drop = nn.Dropout(drop_out)
        
    def dropout(self):
        s_dict= self.att_w1.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.att_w1.load_state_dict(s_dict)

        s_dict= self.att_w2.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.att_w2.load_state_dict(s_dict)

        s_dict= self.att_w3.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.att_w3.load_state_dict(s_dict)

        s_dict= self.att_w4.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.att_w4.load_state_dict(s_dict)

        s_dict=self.C_l.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.C_l.load_state_dict(s_dict)

        s_dict=self.C_a.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.C_a.load_state_dict(s_dict)

        s_dict=self.C_f.state_dict()
        s_dict['weight']=self.drop(s_dict['weight'])
        self.C_f.load_state_dict(s_dict)

        # s_dict=self.W_ac.state_dict()
        # s_dict['weight']=self.drop(s_dict['weight'])
        # self.W_ac.load_state_dict(s_dict)

    def forward(self,h_i):

        wt1=self.s(self.att_w1(h_i))
        wt2=self.s(self.att_w2(h_i))
        wt3=self.s(self.att_w3(h_i))
        wt4=self.s(self.att_w4(h_i))

        h1=torch.mul(h_i,wt1)
        h2=torch.mul(h_i,wt2)
        h3=torch.mul(h_i,wt3)
        h4=torch.mul(h_i,wt4)

        h_l=torch.cat((h1.view(-1,1)[0:self.l_d],h2.view(-1,1)[0:self.l_d],h3.view(-1,1)[0:self.l_d],h4.view(-1,1)[0:self.l_d]),0).view(1,self.num_atten*self.l_d)
        h_a=torch.cat((h1.view(-1,1)[self.l_d:self.l_d+self.a_d],h2.view(-1,1)[self.l_d:self.l_d+self.a_d],h3.view(-1,1)[self.l_d:self.l_d+self.a_d],h4.view(-1,1)[self.l_d:self.l_d+self.a_d]),0).view(1,self.num_atten*self.a_d)
        h_f=torch.cat((h1.view(-1,1)[self.l_d+self.a_d:],h2.view(-1,1)[self.l_d+self.a_d:],h3.view(-1,1)[self.l_d+self.a_d:],h4.view(-1,1)[self.l_d+self.a_d:]),0).view(1,self.num_atten*self.f_d)

        s_l = self.C_l(h_l)
        s_a = self.C_a(h_a)
        s_f = self.C_f(h_f)

        s=torch.cat((s_l.view(-1,1),s_a.view(-1,1),s_f.view(-1,1)),0).view(1,self.l_d+self.a_d+self.f_d)

        z=self.W_ac(s)

        return z 


class MOSI_attention_classifier(nn.Module,ModelIO):

    def init_lstm_param(self,hidden_dim):
        return (variablize(torch.zeros(1, hidden_dim)),variablize(torch.zeros(1, hidden_dim)))

    def init_context_var(self,context_dim):
        return variablize(torch.zeros(1, context_dim))

    def __init__(self,lan_param,audio_param,face_param,num_atten,context_dim,out_dim,drop_out):

        super(MOSI_attention_classifier,self).__init__()

        self.lan_lstm=LSTM_custom(lan_param['input_dim'],lan_param['hidden_dim'],context_dim,drop_out)
        self.audio_lstm=LSTM_custom(audio_param['input_dim'],audio_param['hidden_dim'],context_dim,drop_out)
        self.face_lstm=LSTM_custom(face_param['input_dim'],face_param['hidden_dim'],context_dim,drop_out)
        
        self.num_atten=num_atten
        self.context_dim=context_dim
        self.hidden_comb_dim=lan_param['hidden_dim']+audio_param['hidden_dim']+face_param['hidden_dim']

        self.mab_net = Memory_attention_network(self.hidden_comb_dim,self.context_dim,lan_param['hidden_dim'],audio_param['hidden_dim'],face_param['hidden_dim'],num_atten,drop_out)
        self.W_cout = nn.Linear(self.context_dim,out_dim)

        self.lan_init_param = self.init_lstm_param(lan_param['hidden_dim'])
        self.audio_init_param = self.init_lstm_param(audio_param['hidden_dim']) 
        self.face_init_param = self.init_lstm_param(face_param['hidden_dim'])
        self.z_init = self.init_context_var(self.context_dim)

    def forward(self,opinion):

        for i,x in enumerate(opinion):
            x_lan,x_audio,x_face=filter_train_features(x)
            x_lan=variablize(torch.FloatTensor(x_lan))
            x_audio=variablize(torch.FloatTensor(x_audio))
            x_face=variablize(torch.FloatTensor(x_face))

            if i==0:
                if self.training:
                    self.lan_lstm.dropout()
                    self.audio_lstm.dropout()
                    self.face_lstm.dropout()
                    self.mab_net.dropout()

                h_lan,c_lan=self.lan_lstm.forward(x_lan,self.lan_init_param,self.z_init)
                h_audio,c_audio=self.audio_lstm.forward(x_audio,self.audio_init_param,self.z_init)
                h_face,c_face=self.face_lstm.forward(x_face,self.face_init_param,self.z_init)
                h_i=torch.cat((h_lan,h_audio,h_face),1)      
                z=self.mab_net.forward(h_i)
            else:
                h_lan,c_lan=self.lan_lstm.forward(x_lan,(h_lan,c_lan),z)
                h_audio,c_audio=self.audio_lstm.forward(x_audio,(h_audio,c_audio),z)
                h_face,c_face=self.face_lstm.forward(x_face,(h_face,c_face),z)
                h_i=torch.cat((h_lan,h_audio,h_face),1)
                z=self.mab_net.forward(h_i)

        out = self.W_cout(z)
        return out


def test_model():
    opinion=np.array([[1,2,3,4,5,6,7,8],[3,4,5,6,7,8,9,10]])
    lan_param={'input_dim':3,'hidden_dim':2,'context_dim':2}
    audio_param={'input_dim':3,'hidden_dim':3,'context_dim':2}
    face_param={'input_dim':2,'hidden_dim':1,'context_dim':2}
    context_dim=2
    num_atten=3
    out_dim=1
    drop_out=0.15

    mosi_model=MOSI_attention_classifier(lan_param,audio_param,face_param,num_atten,context_dim,out_dim,drop_out)
    z=mosi_model.forward(opinion)
    print "z",z
    

# test_model()