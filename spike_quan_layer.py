import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
import math
from copy import deepcopy

# torch.set_default_dtype(torch.double)
# torch.set_default_tensor_type(torch.DoubleTensor)

class IFNeuron(nn.Module):
    def __init__(self,q_threshold,level,sym=False):
        super(IFNeuron,self).__init__()
        self.q = 0.0
        self.acc_q = 0.0
        self.q_threshold = q_threshold
        self.is_work = False
        self.cur_output = 0.0
        self.output = 0.0
        # self.steps = torch.tensor(3.0) 
        self.level = torch.tensor(level)
        if sym:
            self.pos_max = torch.tensor(level//2 - 1)
            self.neg_max = torch.tensor(-level//2)
        else:
            self.pos_max = torch.tensor(level - 1)
            self.neg_max = torch.tensor(0)
            
        self.T = 0.0
        self.q_init = False
        self.eps = 0
    
    def reset(self):
        self.q = 0.0
        self.output = 0.0
        self.cur_output = 0.0
        self.acc_q = 0.0
        self.is_work = False
        self.T = 0.0
        self.q_init = False
        self.spike_position = None
        self.neg_spike_position = None

    def forward(self,input):
        x = input/self.q_threshold
        if (not torch.is_tensor(x)) and x == 0.0 and (not torch.is_tensor(self.cur_output)) and self.cur_output == 0.0:
            self.is_work = False
            return x
        
        if not torch.is_tensor(self.cur_output):
            self.cur_output = torch.zeros(x.shape,dtype=x.dtype).to(x.device)
            self.acc_q = torch.zeros(x.shape,dtype=torch.float64).to(x.device)
            self.q = torch.zeros(x.shape,dtype=torch.float64).to(x.device) + 0.5

        self.is_work = True
        
        self.q = self.q + (x.detach().double() if torch.is_tensor(x) else x.double())
        self.acc_q = torch.round(self.acc_q)

        spike_position = (self.q - 1 >= 0) & (self.acc_q < self.pos_max.double())
        neg_spike_position = (self.q < -self.eps) & (self.acc_q > self.neg_max.double())

        self.cur_output[:] = 0
        self.cur_output[spike_position] = 1
        self.cur_output[neg_spike_position] = -1

        self.acc_q = self.acc_q + self.cur_output.double()
        self.q[spike_position] = self.q[spike_position] - 1
        self.q[neg_spike_position] = self.q[neg_spike_position] + 1

        if (x == 0).all() and (self.cur_output==0).all():
            self.is_work = False
        
        # print("self.cur_output",self.cur_output)
        
        return self.cur_output*self.q_threshold

class Spiking_LayerNorm(nn.Module):
    def __init__(self,dim):
        super(Spiking_LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.X = 0.0
        self.Y_pre = None
        
    def forward(self,input):
        self.X = self.X + input
        Y = self.layernorm(self.X)
        if self.Y_pre is not None:
            Y_pre = self.Y_pre.detach().clone()
        else:
            Y_pre = 0.0
        self.Y_pre = Y
        return Y - Y_pre

class spiking_softmax(nn.Module):
    def __init__(self):
        super(spiking_softmax, self).__init__()
        self.X = 0.0
        self.Y_pre = 0.0
    
    def forward(self, input):
        self.X = input + self.X
        Y = F.softmax(self.X,dim=-1)
        Y_pre = deepcopy(self.Y_pre)
        self.Y_pre = Y
        return Y - Y_pre

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class MyQuan(nn.Module):
    def __init__(self,level,sym = False,**kwargs):
        super(MyQuan,self).__init__()
        # self.level_init = level
        self.s_init = 0.0
        self.level = level
        self.sym = sym
        if level >= 256:
            print("level",level)
            self.pos_max = 'full'
        else:
            print("level",level)
            self.pos_max = torch.tensor(level)
            if sym:
                self.pos_max = torch.tensor(float(level//2 - 1))
                self.neg_min = torch.tensor(float(-level//2))
            else:
                self.pos_max = torch.tensor(float(level - 1))
                self.neg_min = torch.tensor(float(0))

        self.s = nn.Parameter(torch.tensor(1.0))
        self.batch_init = 20
        self.init_state = 0
        self.debug = False
        self.tfwriter = None
        self.global_step = 0.0
        self.name = "myquan"

    def __repr__(self):
        return f"MyQuan(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, self.neg_min={self.neg_min})"

    
    def reset(self):
        self.history_max = torch.tensor(0.0)
        self.init_state = 0
        self.is_init = True

    def profiling(self,name,tfwriter,global_step):
        self.debug = True
        self.name = name
        self.tfwriter = tfwriter
        self.global_step = global_step

    def forward(self, x):
        # print("self.pos_max",self.pos_max)
        if self.pos_max == 'full':
            return x
        # print("self.Q_thr in Quan",self.Q_thr,"self.T:",self.T)
        if str(self.neg_min.device) == 'cpu':
            self.neg_min = self.neg_min.to(x.device)
        if str(self.pos_max.device) == 'cpu':
            self.pos_max = self.pos_max.to(x.device)
        min_val = self.neg_min
        max_val = self.pos_max
        # x = F.hardtanh(x, min_val=min_val, max_val=max_val.item())

        # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
        s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)
        # s_grad_scale = s_grad_scale / ((self.level_init)/(self.pos_max))

        # print("self.init_state",self.init_state)
        # print("self.init_state<self.batch_init",self.init_state<self.batch_init)
        # print("self.training",self.training)
        if self.init_state == 0 and self.training:
            self.s.data = torch.tensor(x.detach().abs().mean() * 2 / (self.pos_max.detach().abs().mean() ** 0.5),dtype=torch.float32).cuda()
            self.init_state += 1
        elif self.init_state<self.batch_init and self.training:
            self.s.data = 0.9*self.s.data + 0.1*torch.tensor(torch.mean(torch.abs(x.detach()))*2/(math.sqrt(max_val.detach().abs().mean())),dtype=torch.float32)
            self.init_state += 1
            
        elif self.init_state==self.batch_init and self.training:
            # self.s = torch.nn.Parameter(self.s)
            self.init_state += 1
            print("initialize finish!!!!")

        s_scale = grad_scale(self.s, s_grad_scale)
        # s_scale = s_scale * ((self.level_init)/(self.pos_max))
        output = torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)*s_scale

        if self.debug and self.tfwriter is not None:
            self.tfwriter.add_histogram(tag="before_quan/".format(s_scale.item())+self.name+'_data', values=(x).detach().cpu(), global_step=self.global_step)
            # self.tfwriter.add_histogram(tag="after_clip/".format(s_scale.item())+self.name+'_data', values=(floor_pass(x/s_scale)).detach().cpu(), global_step=self.global_step)
            self.tfwriter.add_histogram(tag="after_quan/".format(s_scale.item())+self.name+'_data', values=((torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))).detach().cpu(), global_step=self.global_step)
            # print("(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val))",(torch.clamp(floor_pass(x/s_scale + 0.5), min=min_val, max=max_val)))
            self.debug = False
            self.tfwriter = None
            self.name = ""
            self.global_step = 0.0
            
        # output = floor_pass(x/s_scale)*s_scale
        return output

class QAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            level = 2,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.level = level

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.quan_q = MyQuan(self.level,sym=True)
        self.quan_k = MyQuan(self.level,sym=True)
        self.quan_v = MyQuan(self.level,sym=True)
        # self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim,bias=False)
        self.quan_proj = MyQuan(self.level,sym=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_quan = MyQuan(self.level,sym=False)
        self.after_attn_quan = MyQuan(self.level,sym=True)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # q, k = self.q_norm(q), self.k_norm(k)
        q = self.quan_q(q)
        k = self.quan_k(k)
        v = self.quan_v(v)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_quan(attn)
        
        attn = self.attn_drop(attn)
        x = attn @ v
        x = self.after_attn_quan(x)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.quan_proj(x)

        return x

def multi(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return x1_sum_t @ x2_t.transpose(-2, -1)  + x1_t @ x2_sum_t.transpose(-2, -1) - x1_t @ x2_t.transpose(-2, -1)

def multi1(x1_t,x2_t,x1_sum_t,x2_sum_t):
    return x1_sum_t @ x2_t + x1_t @ x2_sum_t - x1_t @ x2_t

class SAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            neuron_layer = IFNeuron,
            level = 2,
            
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.neuron_layer = neuron_layer
        self.level = level

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.k_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.v_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        # self.attn_ReLU = nn.ReLU()
        self.attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=False)
        self.after_attn_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.proj = nn.Linear(dim, dim,bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_IF = self.neuron_layer(q_threshold=torch.tensor(1.0),level=self.level,sym=True)
        self.Ssoftmax = spiking_softmax()
        self.T = 0

    def reset(self):
        self.q_IF.reset()
        self.k_IF.reset()
        self.q_IF.reset()
        self.attn_IF.reset()
        self.after_attn_IF.reset()
        self.proj_IF.reset()
        self.T = 0

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.q_IF(q)
        k = self.k_IF(k)
        v = self.v_IF(v)
        
        q = q * self.scale
        q_acc = self.q_IF.acc_q * self.scale * self.q_IF.q_threshold
        attn = multi(q,k,q_acc,self.k_IF.acc_q*self.k_IF.q_threshold)
        attn = self.Ssoftmax(attn)
        attn = self.attn_IF(attn)

        attn = self.attn_drop(attn)

        x = multi1(attn,v,self.attn_IF.acc_q*self.attn_IF.q_threshold,self.v_IF.acc_q*self.v_IF.q_threshold)

        x = self.after_attn_IF(x)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.proj_IF(x)

        self.T = self.T + 1

        return x
class SpikeMaxPooling(nn.Module):
    def __init__(self,maxpool):
        super(SpikeMaxPooling,self).__init__()
        self.maxpool = maxpool
        
        self.accumulation = None
    
    def reset(self):
        self.accumulation = None

    def forward(self,x):
        old_accu = self.accumulation
        if self.accumulation is None:
            self.accumulation = x
        else:
            self.accumulation = self.accumulation + x
        
        if old_accu is None:
            output = self.maxpool(self.accumulation)
        else:
            output = self.maxpool(self.accumulation) - self.maxpool(old_accu)

        # print("output.shape",output.shape)
        # print(output[0][0][0:4][0:4])
        
        return output
    
class LLConv2d(nn.Module):
    def __init__(self,conv,**kwargs):
        super(LLConv2d,self).__init__()
        self.conv = conv
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = self.level
        self.realize_time = self.steps
        
        
    def reset(self):
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self,input):
        # print("LLConv2d.steps",self.steps)
        x = input
        N,C,H,W = x.shape
        F_h,F_w = self.conv.kernel_size
        S_h,S_w = self.conv.stride
        P_h,P_w = self.conv.padding
        C = self.conv.out_channels
        H = math.floor((H - F_h + 2*P_h)/S_h)+1
        W = math.floor((W - F_w + 2*P_w)/S_w)+1

        if self.zero_output is None:
            # self.zero_output = 0.0
            self.zero_output = torch.zeros(size=(N,C,H,W),device=x.device,dtype=x.dtype)

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x==0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/self.steps if self.conv.bias is not None else 0.0)
                self.realize_time = self.realize_time - 1
                self.is_work = True
                return output
            return self.zero_output

        output = self.conv(x)

        if self.neuron_type == 'IF':
            pass
        else:
            if self.conv.bias is None:
                pass
            else:
                # if not self.first:
                #     output = output - self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                output = output - (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) if self.conv.bias is not None else 0.0)
                if self.realize_time > 0:
                    output = output + (self.conv.bias.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/self.steps if self.conv.bias is not None else 0.0)
                    self.realize_time = self.realize_time - 1
                    # print("conv2d self.realize_time",self.realize_time)
                    

        self.is_work = True
        self.first = False

        return output

class LLLinear(nn.Module):
    def __init__(self,linear,**kwargs):
        super(LLLinear,self).__init__()
        self.linear = linear
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = kwargs["neuron_type"]
        self.level = kwargs["level"]
        self.steps = self.level
        self.realize_time = self.steps
    def reset(self):
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self,input):
        # print("LLLinear.steps",self.steps)
        x = input
        # if x.ndim == 2:
        #     B,N = x.shape
        # elif x.ndim == 3:
        #     B,C,N = x.shape
        # N = self.linear.out_features
        if self.zero_output is None:
            self.zero_output = torch.zeros(size=x.shape,device=x.device,dtype=x.dtype)

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x==0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (self.linear.bias.data.unsqueeze(0)/self.steps if self.linear.bias is not None else 0.0)
                self.realize_time = self.realize_time - 1
                self.is_work = True
                return output
            return self.zero_output

        output = self.linear(x)

        if self.neuron_type == 'IF':
            pass
        else:
            if self.linear.bias is None:
                pass
            else:
                # if not self.first:
                #     output = output - self.linear.bias.data.unsqueeze(0)
                output = output - (self.linear.bias.data.unsqueeze(0) if self.linear.bias is not None else 0.0)
                if self.realize_time > 0:
                    output = output + (self.linear.bias.data.unsqueeze(0)/self.steps if self.linear.bias is not None else 0.0)
                    self.realize_time = self.realize_time - 1
                    # print("linear self.realize_time",self.realize_time)


        self.is_work = True
        self.first = False

        return output
