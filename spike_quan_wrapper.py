
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from spike_quan_layer import MyQuan,IFNeuron,LLConv2d,LLLinear,SpikeMaxPooling,QAttention,SAttention,spiking_softmax,Spiking_LayerNorm
import sys
from timm.models.vision_transformer import Attention,Mlp

def get_subtensors(tensor,mean,std,sample_grain=255,output_num=4):
    for i in range(int(sample_grain)):
        output = (tensor/sample_grain).unsqueeze(0)
        # output = (tensor).unsqueeze(0)
        if i == 0:
            accu = output
        else:
            accu = torch.cat((accu,output),dim=0)
    return accu


class Judger():
	def __init__(self):
		self.network_finish=True

	def judge_finish(self,model):
		children = list(model.named_children())
		for name, child in children:
			is_need = False
			if isinstance(child, IFNeuron) or isinstance(child, LLLinear) or isinstance(child, LLConv2d):
				self.network_finish = self.network_finish and (not model._modules[name].is_work)
				# print("child",child,"network_finish",self.network_finish,"model._modules[name].is_work",(model._modules[name].is_work))
				is_need = True
			if not is_need:
				self.judge_finish(child)

	def reset_network_finish_flag(self):
		self.network_finish = True

def attn_convert(QAttn:QAttention,SAttn:SAttention,level):
    SAttn.qkv = LLLinear(linear = QAttn.qkv,neuron_type = "ST-BIF",level = level)
    SAttn.proj = LLLinear(linear = QAttn.proj,neuron_type = "ST-BIF",level = level)
    SAttn.q_IF.q_threshold = QAttn.quan_q.s.data
    SAttn.k_IF.q_threshold = QAttn.quan_k.s.data
    SAttn.v_IF.q_threshold = QAttn.quan_v.s.data
    SAttn.attn_IF.q_threshold = QAttn.attn_quan.s.data
    SAttn.after_attn_IF.q_threshold = QAttn.after_attn_quan.s.data
    SAttn.proj_IF.q_threshold = QAttn.quan_proj.s.data


class SNNWrapper(nn.Module):
    
    def __init__(self, ann_model, cfg, time_step = 2000,Encoding_type="analog",**kwargs):
        super(SNNWrapper, self).__init__()
        self.T = time_step
        self.cfg = cfg
        self.finish_judger = Judger()
        self.Encoding_type = Encoding_type
        self.level = kwargs["level"]
        self.neuron_type = kwargs["neuron_type"]
        self.model = ann_model
        self.kwargs = kwargs
        self.model_name = kwargs["model_name"]
        self._replace_weight(self.model)
    
    def _replace_weight(self,model):
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                SAttn = SAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=self.level)
                attn_convert(QAttn=child,SAttn=SAttn,level=self.level)
                model._modules[name] = SAttn
                is_need = True
            elif isinstance(child, nn.Conv2d):
                model._modules[name] = LLConv2d(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.Linear):
                model._modules[name] = LLLinear(child,**self.kwargs)
                is_need = True
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = SpikeMaxPooling(child)
                is_need = True
            elif isinstance(child, nn.LayerNorm):
                SNN_LN = Spiking_LayerNorm(child.weight.shape[0])
                SNN_LN.layernorm.weight.data = child.weight.data
                SNN_LN.layernorm.bias.data = child.bias.data                
                model._modules[name] = SNN_LN
                is_need = True
            elif isinstance(child, MyQuan):
                neurons = IFNeuron(q_threshold = torch.tensor(1.0),sym=child.sym,level = child.pos_max.double())
                neurons.q_threshold=child.s.data
                neurons.neuron_type=self.neuron_type
                neurons.steps = child.pos_max.double()
                neurons.is_init = False                    
                model._modules[name] = neurons     
                is_need = True
            elif isinstance(child, nn.ReLU):
                model._modules[name] = nn.Identity()
                is_need = True
            if not is_need:            
                self._replace_weight(child)

    def forward(self,x, verbose=False):
        accu = None
        count1 = 0
        accu_per_timestep = []
        # print("self.bit",self.bit)
        # x = x*(2**self.bit-1)+0.0

        if self.Encoding_type == "rate":
            self.mean = 0.0
            self.std  = 0.0
            x = get_subtensors(x,self.mean,self.std,sample_grain=self.level)
            # print("x.shape",x.shape)
        while(1):
            self.finish_judger.reset_network_finish_flag()
            self.finish_judger.judge_finish(self)
            network_finish = self.finish_judger.network_finish
            # print(f"==================={count1}===================")
            if (count1 > 0 and network_finish) or count1 >= self.T:
                break
            # if self.neuron_type.count("QFFS") != -1 or self.neuron_type == 'ST-BIF':
            if self.model_name.count("VIT")>0 and self.count1 > 0:
                self.model.pos_embed = 0.0
            if self.Encoding_type == "rate":
                if count1 < x.shape[0]:
                    input = x[count1]
                else:
                    input = torch.zeros(x[0].shape).to(x.device)            
            else:
                if count1 == 0:
                    input = x
                else:
                    input = torch.zeros(x.shape).to(x.device)
            # elif self.neuron_type == 'IF':
            #     input = x
            # else:
            #     print("No implementation of neuron type:",self.neuron_type)
            #     sys.exit(0)

            output = self.model(input)
            # print("output",output.sum())
            
            if count1 == 0:
                accu = output+0.0
            else:
                accu = accu+output
            if verbose:
                accu_per_timestep.append(accu)
            # print("accu",accu.sum(),"output",output.sum())
            count1 = count1 + 1
            if count1 % 100 == 0:
                print(count1)

        # print("verbose",verbose)
        if verbose:
            accu_per_timestep = torch.stack(accu_per_timestep,dim=0)
            return accu,count1,0,accu_per_timestep
        else:
            return accu,count1,0        


def myquan_replace(model,level):
    index = 0
    cur_index = 0
    def get_index(model):
        nonlocal index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, QAttention):
                index = index + 1
                is_need = True
            if not is_need:
                get_index(child)

    def _myquan_replace(model,level):
        nonlocal index
        nonlocal cur_index
        children = list(model.named_children())
        for name, child in children:
            is_need = False
            if isinstance(child, Attention):
                # print(children)
                qattn = QAttention(dim=child.num_heads*child.head_dim,num_heads=child.num_heads,level=level)
                qattn.qkv = child.qkv
                # qattn.q_norm = child.q_norm
                # qattn.k_norm = child.k_norm
                qattn.attn_drop = child.attn_drop
                qattn.proj = child.proj
                qattn.proj_drop = child.proj_drop
                model._modules[name] = qattn
                print("index",cur_index,"myquan replace finish!!!!")
                cur_index = cur_index + 1
                is_need = True
            elif isinstance(child,Mlp):
                model._modules[name].act = nn.Sequential(MyQuan(level,sym = False),child.act)
                model._modules[name].fc2 = nn.Sequential(child.fc2,MyQuan(level,sym = True))                
                is_need = True
            elif isinstance(child, nn.Conv2d):
                model._modules[name] = nn.Sequential(child,MyQuan(level,sym = True))
                is_need = True
            if not is_need:
                _myquan_replace(child,level)
    get_index(model)
    _myquan_replace(model,level)




