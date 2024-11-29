import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FunctionalParamVectorWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super(FunctionalParamVectorWrapper, self).__init__()
        self.module = module

    def forward(self, param_vector: torch.Tensor, raw: torch.Tensor, embed: torch.Tensor, *args, **kwargs):
        params = {}
        start = 0

        if isinstance(self.module, DynamicHyperNet):
            considered = self.module.weight_generator.named_parameters()
        else:
            considered = self.module.named_parameters()
        
        for name, p in list(considered):
            end = start + np.prod(p.size())
            params[name] = param_vector[start:end].view(p.size())
            start = end

        if isinstance(self.module, DynamicHyperNet):
            out = torch.func.functional_call(self.module.weight_generator, params, (embed,))
            return self.module.propagate(out, raw, embed, *args, **kwargs)
        else:
            out = torch.func.functional_call(self.module, params, (raw, *args), kwargs)
            return out

class DynamicHyperNet(nn.Module):
    def __init__(self, target_network, device="cpu"):
        super().__init__()
        self.device = device
        
        if isinstance(target_network, DynamicHyperNet):
            self.num_params_to_estimate = target_network.num_weight_gen_params
        else:
            self.num_params_to_estimate = int(sum(p.numel() for p in target_network.parameters()))

        self.create_params()

        self.num_weight_gen_params = sum(p.numel() for p in self.weight_generator.parameters())
        self.target_param_updater = FunctionalParamVectorWrapper(target_network)
    
    def propagate_forward(self, raw, embed, *args, **kwargs):
        out = self.weight_generator.forward(embed)
        return self.propagate(out, raw, embed, *args, **kwargs)
    
    def propagate(self, out, raw, embed, *args, **kwargs):
        return self.target_param_updater(out.view(-1), raw, embed, *args, **kwargs)

    def create_params(self):
        raise NotImplementedError("Subclasses implement this method to initialize the weight generator.")
    
class DynamicEmbedding(nn.Module):
    def __init__(self, top_hypernet, batch_size, num_heads, dropout=0.1):
        super().__init__()

        self.batch_size = batch_size
        self.create_params()

        self.num_weight_gen_params = sum(p.numel() for p in self.weight_generator.parameters())
        self.top_hypernet = top_hypernet
    
    def create_params(self):
        raise NotImplementedError("Subclasses implement this method to initialize the weight generator.")
    
    def embed_and_propagate(self, raw):
        assert raw.shape[0] <= self.batch_size
        
        padded = raw
        diff = self.batch_size - padded.shape[0]
        
        if padded.shape[0] < self.batch_size:
            num_no_pads = np.array([[0, 0] for _ in range(padded.ndim - 1)])
            num_no_pads = num_no_pads.flatten()
            padded = F.pad(padded, (*num_no_pads, 0, diff))
        
        embed = self.weight_generator.forward(padded, diff)
        return self.top_hypernet.propagate_forward(raw, embed)
