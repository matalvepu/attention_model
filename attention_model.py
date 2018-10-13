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
        if torch.isnan(h_[0][0]):
            print "h nan",h_
            print "x",x 
            # print "self.W_xi",self.W_xi.weight
            # print self.W_hi.weight 
            # print self.W_xf.weight  
            # print self.W_hf.weight 
            # print self.W_zf.weight  
            # print self.W_xg.weight  
            # print self.W_hg.weight  
            # print self.W_zg.weight  
            # print self.W_xo.weight  
            # print self.W_ho.weight  
            # print self.W_zo.weight  
            # print self.W_zi.weight 

        return h_,c_


class Memory_attention_network(nn.Module):

    def __init__(self,hidden_comb_dim,context_dim,num_atten):

        super(Memory_attention_network,self).__init__()
        self.hidden_comb_dim=hidden_comb_dim
        self.num_atten=num_atten
        #self.attention_weigths=[nn.Linear(context_dim,hidden_comb_dim).cuda() for i in range(num_atten)]
        self.att_w1=nn.Linear(context_dim,hidden_comb_dim)
    	self.att_w2=nn.Linear(context_dim,hidden_comb_dim)
    	self.att_w3=nn.Linear(context_dim,hidden_comb_dim)
    	self.W_ac = nn.Linear(hidden_comb_dim,context_dim)
        self.s=nn.LogSoftmax()

    def forward(self,h_i,z):
	#print "z in",z
        h=variablize(torch.zeros(self.hidden_comb_dim).view(1,-1))
        wt1=self.s(self.att_w1(z))
        wt2=self.s(self.att_w2(z))
        wt3=self.s(self.att_w3(z))
        h+=torch.mul(h_i,wt1)
        h+=torch.mul(h_i,wt2)
        h+=torch.mul(h_i,wt3)
        #for i in range(self.num_atten):
        #    print "self atten", self.attention_weigths[i].weight.device
        #    w_t=self.s(self.attention_weigths[i](z))
        #    h_t=torch.mul(h_i,w_t)
        #    h+=h_t
        #print "h",h.device
        z=self.W_ac(h)

        return z 


class MOSI_attention_classifier(nn.Module,ModelIO):

    def init_lstm_param(self,hidden_dim):
        return (variablize(torch.zeros(1, hidden_dim)),variablize(torch.zeros(1, hidden_dim)))

    def init_context_var(self,context_dim):
        return variablize(torch.zeros(1, context_dim))

    def __init__(self,lan_param,audio_param,face_param,num_atten,context_dim,out_dim):

        super(MOSI_attention_classifier,self).__init__()

        self.lan_lstm=LSTM_custom(lan_param['input_dim'],lan_param['hidden_dim'],context_dim)
        #self.audio_lstm=LSTM_custom(audio_param['input_dim'],audio_param['hidden_dim'],context_dim)
        self.face_lstm=LSTM_custom(face_param['input_dim'],face_param['hidden_dim'],context_dim)
        
        self.num_atten=num_atten
        self.context_dim=context_dim
        self.hidden_comb_dim=lan_param['hidden_dim']+face_param['hidden_dim']

        self.mab_net = Memory_attention_network(self.hidden_comb_dim,self.context_dim,self.num_atten)
        self.W_cout = nn.Linear(self.context_dim,out_dim)

        self.lan_init_param = self.init_lstm_param(lan_param['hidden_dim'])
        #self.audio_init_param = self.init_lstm_param(audio_param['hidden_dim']) 
        self.face_init_param = self.init_lstm_param(face_param['hidden_dim'])
        # self.W_ac = nn.Linear(self.hidden_comb_dim,context_dim)
        # self.sig=nn.Sigmoid()        


    def forward(self,opinion):
	#print self.lan_lstm.W_xi.weight.device
        #print self.mab_net.W_ac.weight.device
        # s=nn.Softmax()
        for i,x in enumerate(opinion):
            x_lan,x_audio,x_face=filter_train_features(x)
            x_lan=variablize(torch.FloatTensor(x_lan))
            # x_audio=variablize(torch.FloatTensor(x_audio))
            x_face=variablize(torch.FloatTensor(x_face))

            #lhtmstep 
            if i==0:
                if self.training:
                    self.lan_lstm.dropout()
                    # self.audio_lstm.dropout()
                    self.face_lstm.dropout()

                z_init = self.init_context_var(self.context_dim)
                h_lan,c_lan=self.lan_lstm.forward(x_lan,self.lan_init_param,z_init)
                #h_audio,c_audio=self.audio_lstm.forward(x_audio,self.audio_init_param,z_init)
                h_face,c_face=self.face_lstm.forward(x_face,self.face_init_param,z_init)
                h_i=torch.cat((h_lan,h_face),1)           
                z=self.mab_net.forward(h_i,z_init)
            else:
                h_lan,c_lan=self.lan_lstm.forward(x_lan,(h_lan,c_lan),z)
                #h_audio,c_audio=self.audio_lstm.forward(x_audio,(h_audio,c_audio),z)
                h_face,c_face=self.face_lstm.forward(x_face,(h_face,c_face),z)
                h_i=torch.cat((h_lan,h_face),1)
                z=self.mab_net.forward(h_i,z)

            # z=self.W_ac(h_i)
            # print "z",z
            # print "x",x
            # print "h_lan",h_lan
            # print "h_audio",h_audio
            # print "h_face",h_face

            # print "####DEBUG#####", i
            # print "h_lan",h_lan
            # print "h_face",h_face
            # print "h_i",h_i
            # print "atten_weights",attn_weigths
            # print "h_atten",h_atten
            # print "z",z 
	#print "z",z
       
        out = self.W_cout(z)
        # print "out", out
        # print "out",out
        return out


# class MOSI_attention_classifier(nn.Module,ModelIO):

#     def init_lstm_param(self,hidden_dim):
#         return (variablize(torch.zeros(1, hidden_dim)),variablize(torch.zeros(1, hidden_dim)))

#     def init_context_var(self,context_dim):
#         return variablize(torch.zeros(1, context_dim))

#     def __init__(self,lan_param,audio_param,face_param,num_atten,context_dim,out_dim):
        
#         super(MOSI_attention_classifier,self).__init__()

#         self.lan_lstm=LSTM_custom(lan_param['input_dim'],lan_param['hidden_dim'],context_dim)
#         self.audio_lstm=LSTM_custom(audio_param['input_dim'],audio_param['hidden_dim'],context_dim)
#         self.face_lstm=LSTM_custom(face_param['input_dim'],face_param['hidden_dim'],context_dim)
        
#         self.num_atten=num_atten
#         self.context_dim=context_dim

#         self.hidden_comb_dim=lan_param['hidden_dim']+audio_param['hidden_dim']+face_param['hidden_dim']
#         self.W_a = nn.Linear(self.context_dim,self.hidden_comb_dim)
#         self.lan_init_param = self.init_lstm_param(lan_param['hidden_dim'])
#         self.audio_init_param = self.init_lstm_param(audio_param['hidden_dim'])
#         self.face_init_param = self.init_lstm_param(face_param['hidden_dim'])      
#         self.W_ac = nn.Linear(self.hidden_comb_dim,self.context_dim)
#         self.W_cout = nn.Linear(self.context_dim,out_dim)
#         self.z_init = self.init_context_var(self.context_dim)
#         # print "z_init",self.z_init

#     def forward(self,opinion):
#         s=nn.Softmax()
#         for i,x in enumerate(opinion):
#             x_lan,x_audio,x_face=filter_train_features(x)
#             x_lan=variablize(torch.FloatTensor(x_lan))
#             x_audio=variablize(torch.FloatTensor(x_audio))
#             x_face=variablize(torch.FloatTensor(x_face))

#             #lhtmstep 
#             if i==0:
#                 if self.training:
#                     self.lan_lstm.dropout()
#                     self.audio_lstm.dropout()
#                     self.face_lstm.dropout()

#                 h_lan,c_lan=self.lan_lstm.forward(x_lan,self.lan_init_param,self.z_init)
#                 h_audio,c_audio=self.audio_lstm.forward(x_audio,self.audio_init_param,self.z_init)
#                 h_face,c_face=self.face_lstm.forward(x_face,self.face_init_param,self.z_init)
               
#                 h_i=torch.cat((h_lan,h_audio,h_face),1)
#                 attn_weigths=s(self.W_a(self.z_init))
#                 h_atten = torch.mul(h_i,attn_weigths)
#                 z=self.W_ac(h_atten)
#             else:
#                 h_lan,c_lan=self.lan_lstm.forward(x_lan,(h_lan,c_lan),z)
#                 print("audio forward")
#                 h_audio,c_audio=self.audio_lstm.forward(x_audio,(h_audio,c_audio),z)
#                 h_face,c_face=self.face_lstm.forward(x_face,(h_face,c_face),z)
#                 h_i=torch.cat((h_lan,h_audio,h_face),1)
#                 attn_weigths=s(self.W_a(z))
#                 h_atten = torch.mul(h_i,attn_weigths)
#                 z=self.W_ac(h_atten)

#             # print "####DEBUG#####", i
#             # print "h_lan",h_lan
#             # print "h_face",h_face
#             # print "h_i",h_i
#             # print "atten_weights",attn_weigths
#             # print "h_atten",h_atten
#             # print "z",z 

#         out = self.W_cout(z)
#         # print "out", out
#         return out

def test_model():
    opinion=np.array([[1,2,3,4,5,6,7,8],[3,4,5,6,7,8,9,10]])
    lan_param={'input_dim':3,'hidden_dim':2,'context_dim':2}
    audio_param={'input_dim':3,'hidden_dim':2,'context_dim':2}
    face_param={'input_dim':2,'hidden_dim':2,'context_dim':2}
    context_dim=2
    num_atten=2
    out_dim=1

    mosi_model=MOSI_attention_classifier(lan_param,audio_param,face_param,num_atten,context_dim,out_dim)
    # print mosi_model
    print mosi_model.forward(opinion)



# test_model()

