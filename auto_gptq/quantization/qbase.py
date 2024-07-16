from abc import ABC
from typing import List, Dict
import torch
import torch.nn as nn

def find_module(m: nn.Module, name:str) -> nn.Module:
    if '.' in name:
        m_ = m
        for n in name.split('.'):
            m_ = getattr(m_, n)
        return m_
    else:
        return getattr(m, name)

class QBase(ABC):
    def __init__(self, layers: nn.ModuleList, 
                 inside_layer_modules=List[List[str]], 
                 **kwargs):
        self.layers = layers
        self.inside_layers_modules = inside_layer_modules
        self.hooks = []
        self.inputs = None
        self.attention_masks = None
        self.position_ids = None
        self.input_kwargs = None

    def set_inputs(self, inputs: List[List[torch.Tensor]],
                   attention_masks: List[torch.Tensor],
                   position_ids: List[torch.Tensor],
                   layer_input_kwargs: Dict[str, torch.Tensor]):
        '''force to cpu to save memory'''
        assert str(inputs[0][0].device) == 'cpu'
        assert str(attention_masks[0].device) == 'cpu'
        assert str(position_ids[0].device) == 'cpu'

        self.inputs = inputs
        self.attention_masks = attention_masks
        self.position_ids = position_ids
        self.input_kwargs = layer_input_kwargs

    def forward_collect(self):
        for inp, attn, pos, kwargs in zip(self.inputs, self.attention_masks, self.position_ids, self.input_kwargs):
            layer_input = []
            for k, layer_inp in enumerate(inp):
                layer_input.append(layer_inp.cuda())

            layer_attention_mask = attn.cuda()
            additional_layer_inputs = {"attention_mask": layer_attention_mask}
            layer_position_ids = (
                None if pos is None else pos.cuda()
            )
            if layer_position_ids is not None:
                additional_layer_inputs["position_ids"] = layer_position_ids
            for k, v in kwargs.items():
                additional_layer_inputs[k] = v if not isinstance(v, torch.Tensor) else v.cuda()
            for layer in self.layers:
                layer_input = [layer(*layer_input, **additional_layer_inputs)]

    @property
    def quant_stages(self):
        raise NotImplementedError

    def register_forward_hooks(self, m: nn.Module, name: str):
        raise NotImplementedError

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def quant(self):
        self.layers.cuda()
        for stages in self.quant_stages:
            for n in stages:
                m = find_module(self.layers, n)
                self.register_forward_hooks(m, n)
            self.forward_collect()
            self.remove_hooks()
            self.do_quant()
            self.free()
        self.layers.cpu()
    
    def do_quant(self):
        raise NotImplementedError

    def free(self):
        raise NotImplementedError