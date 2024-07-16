import torch
import torch.nn as nn
import math
from logging import getLogger

from torch.autograd import Function
from typing import List
from ..qbase import QBase
from ..utils import round_ste

logger = getLogger(__name__)

class WFunc(Function):
    @staticmethod
    def forward(ctx, scale, bias, w_int, w_org, L, nbits):
        w_hat = scale * w_int + bias
        ctx.save_for_backward(scale, bias, w_int, w_org, L, w_hat)
        ctx.nbits = nbits
        return w_hat

    @staticmethod
    def backward(ctx, grad_output):
        scale, bias, w_int, w_org, L, w_hat = ctx.saved_tensors
        nbits = ctx.nbits

        B = w_int - (w_org+(w_org-w_hat)@L-bias)/scale
        A = (L + torch.eye(L.shape[0], device=L.device))
        X = torch.linalg.solve_triangular(A, B, upper=True, left=False)
        grad_scale = X.mul_(grad_output)
        grad_bias = torch.zeros_like(w_int).sum(dim=1, keepdim=True)

        grad_bias[w_int==0] = 1.0
        grad_bias[w_int==2**nbits-1] = 1.0
        grad_bias = grad_bias.mul_(grad_output).sum(dim=1, keepdim=True)

        return grad_scale, grad_bias, None, None, None, None

    
class pLinear(nn.Module):
    def __init__(self, nbits: int, W: torch.Tensor, H: torch.Tensor):
        super(pLinear, self).__init__()
        self.nbits = nbits
        self.register_buffer('W', W.cpu())
        self.register_buffer('H', H.cpu())
        self.register_buffer('Wint', torch.zeros_like(W, device='cpu'))
        h, _ = W.shape
        self.log_scale = nn.Parameter(torch.zeros(h, device='cpu'))
        self.bias  = nn.Parameter(torch.zeros(h, device='cpu'))

    def init(self):
        self.ldl_init()
        self.param_init()
        self.quip_solver()
    
    def ldl_init(self):
        percdamp = 0.01
        H = self.H.clone()
        H = torch.flip(H)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        L = torch.linalg.cholesky(H)
        L = torch.flip(L, [0,1])
        L = L @ torch.diag(1/L.diag())
        L = L - torch.eye(L.shape[0], device=L.device)
        self.register_buffer('L', L)

    @torch.no_grad
    def quip_iter(self):
        a = self.log_scale.unsqueeze(1).exp()
        b = self.bias.unsqueeze(1)
        w_hat = a * self.Wint + b
        return torch.clamp(
            round_ste(
                (self.W - (self.W - w_hat) @ self.L - b) / a
                ),
            min=0, 
            max=2**self.nbits-1
        )
    
    def param_init(self):
        w_max, _ = self.W.max(1)
        w_min, _ = self.W.min(1)
        self.log_scale = ((w_max - w_min)/(2**self.nbits-1)).log()
        self.bias = w_min
        a = self.log_scale.unsqueeze(1).exp()
        b = self.bias.unsqueeze(1)
        self.Wint = torch.clamp(
            round_ste((self.W - b) / a),
            min=0, 
            max=2**self.nbits-1
        )

    def quip_solver(self):
        err = 1e8
        thre = self.Wint.nelement()/1000
        while err > thre:
            w = self.quip_iter()
            err = ((w - self.Wint) != 0).sum()
            self.Wint = w

    def forward(self, input):
        w = WFunc.apply(
            self.log_scale.exp().unsqueeze(1),
            self.bias,
            self.Wint,
            self.W,
            self.L,
            self.nbits
        )
        return input@w.T

class BpGPTQ(QBase):
    def __init__(self, layers: nn.ModuleList, 
                 inside_layer_modules=List[List[str]], 
                 **kwargs):
        super(BpGPTQ, self).__init__(layers,
                                         inside_layer_modules, 
                                         **kwargs)
        assert len(self.layers) == 1
        self.nbits = kwargs['nbits']
        self.H = dict()
        self.nsamples = dict()
        self.inps = dict()

    @property
    def quant_stages(self):
        stages = []
        for ts in self.inside_layers_modules:
            for t in ts:
                stages.append(['0.'+t])
        return stages

    def register_forward_hooks(self, m: nn.Module, name):
        self.H[name] = 0
        self.nsamples[name] = 0
        self.inps[name] = []

        def add_batch(_, inp, out):
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
                self.inps[name].append(inp)
            inp = inp.t()

            self.H[name] *= self.nsamples[name] / (self.nsamples[name] + tmp)
            self.nsamples[name] += tmp
            inp = math.sqrt(2 / self.nsamples[name]) * inp.float()
            self.H[name] += inp.matmul(inp.t())
        self.hooks.append(m.register_forward_hook(add_batch))

    def do_quant(self):
        for n in self.gptq.keys():
            logger.info(f"Quantizing {n} ...")
            self.gptq[n].fasterquant(
                group_size=-1,
            )
        
    def free(self):
        self.gptq = dict()
        self.H = dict()
        self.nsamples = dict()
        self.inps = dict()
        torch.cuda.empty_cache()

__all__ = ["WapperGPTQ"]