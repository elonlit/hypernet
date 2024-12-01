import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseNet(nn.Module):
    def __init__(self, num_backward_connections=0, connection_type="avg", device="cpu"):
        super().__init__()
        self.num_backward_connections = num_backward_connections
        self.connection_type = connection_type
        self.device = device

    def create_params(self):
        raise NotImplementedError("Subclasses implement this method to initialize the weight generator network.")

class HyperNet(BaseNet):
    """General HyperNetwork class"""
    def __init__(self, target_network, num_backward_connections=0, connection_type="avg", device="cpu"):
        super().__init__(num_backward_connections, connection_type, device)
        self._target_network = target_network

class SharedEmbeddingUpdateParams(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, prev_params, param_vector: torch.Tensor, raw: torch.Tensor, embed: torch.Tensor, *args, **kwargs):
        if self.module.connection_type == "avg":
            pool_layer = nn.AdaptiveAvgPool1d(param_vector.shape[0])
        elif self.module.connection_type == "max":
            pool_layer = nn.AdaptiveMaxPool1d(param_vector.shape[0])
        else:
            raise ValueError("Invalid connection type")

        addition_vectors = []
        # connect `min(num_backward_connections, len(prev_params))` previous params to the current ones
        for i in range(max(len(prev_params) - self.module.num_backward_connections, 0), len(prev_params)):
            addition_vector = prev_params[i].view(1, -1)
            addition_vector = pool_layer(addition_vector).view(-1)
            addition_vectors.append(addition_vector)
        
        params = {}
        start = 0
        for name, p in self.module.weight_generator.named_parameters():
            end = start + np.prod(p.size())
            params[name] = param_vector[start:end].view(p.size())
            for addition_vector in addition_vectors:
                params[name] += addition_vector[start:end].view(p.size())
            start = end
        
        if isinstance(self.module, SharedEmbeddingHyperNet):
            assert embed.shape == (self.module.num_embeddings, *self.module.embedding_dim)

            out = torch.func.functional_call(self.module.weight_generator, params, (embed,))
            return self.module.propagate(prev_params, out, raw, embed, *args, **kwargs)
        elif isinstance(self.module, BaseNet):
            assert raw.shape[1:] == self.module.input_dim

            out = torch.func.functional_call(self.module.weight_generator, params, (raw, *args), kwargs)
            return out

class SharedEmbeddingHyperNet(HyperNet):
    """
    If the input data is of shape (B, *S), each HyperNet is provided an `embed` from `SharedEmbedding`
    of shape (1, *S). For all n >= 1, H_n outputs a tensor of shape (P_(n-1),) where P_(n-1) is the
    number of parameters in the (n-1)th HyperNet. The output network is fed in the full data,
    from which it makes a prediction.
    """
    def __init__(self, target_network, num_backward_connections=0, connection_type="avg", device="cpu"):
        super().__init__(target_network, num_backward_connections, connection_type, device)
        self.target_param_updater = SharedEmbeddingUpdateParams(target_network)

    def propagate_forward(self, raw, embed, *args, **kwargs):
        assert embed.shape == (self.num_embeddings, *self.embedding_dim)
        
        out = self.weight_generator.forward(embed)
        return self.propagate([], out, raw, embed, *args, **kwargs)
    
    def propagate(self, prev_params, out, raw, embed, *args, **kwargs):
        prev_params.append(torch.nn.utils.parameters_to_vector(self.weight_generator.parameters()))
        return self.target_param_updater(prev_params, out.view(-1), raw, embed, *args, **kwargs)

class SharedEmbedding(nn.Module):
    def __init__(self, top_hypernet, num_embeddings):
        super().__init__()
        self.top_hypernet = top_hypernet
        self.num_embeddings = num_embeddings
    
    def create_params(self):
        raise NotImplementedError("Subclasses implement this method to initialize the weight generator network.")

    @staticmethod
    def build_weight_generator(module):
        module.create_params()
        module.num_weight_gen_params = sum(p.numel() for p in module.weight_generator.parameters())
    
    def build(self, input_shape, embedding_dim):
        if isinstance(embedding_dim, int):
            embedding_dim = (embedding_dim,)
        
        self.embedding_dim = embedding_dim
        SharedEmbedding.build_weight_generator(self)

        _hypernet_stack = []
        curr_hypernet = self.top_hypernet
        while isinstance(curr_hypernet, SharedEmbeddingHyperNet):
            _hypernet_stack.append(curr_hypernet)

            curr_hypernet.num_embeddings = self.num_embeddings
            curr_hypernet.embedding_dim = self.embedding_dim
            curr_hypernet = curr_hypernet._target_network
        
        curr_hypernet.input_dim = input_shape[1:]
        _hypernet_stack.append(curr_hypernet)

        while _hypernet_stack:
            curr_hypernet = _hypernet_stack.pop()

            if isinstance(curr_hypernet, SharedEmbeddingHyperNet):
                curr_hypernet.num_params_to_estimate = curr_hypernet._target_network.num_weight_gen_params

            SharedEmbedding.build_weight_generator(curr_hypernet)

# --------------------------------------------------------------------------------
#             SHARED EMBEDDING DYNAMIC HYPERNETWORK ARCHITECTURE
# --------------------------------------------------------------------------------
#             x ------> [embed layer] ------> embed (batch size compressed)
#             |                                          /   |   \
#             |                                       [hypernetworks] ----\
#             |                                              |   ^--------/
#             |                                              |
#             |                                              |
#             \-------------> [output network] <-------------/
# --------------------------------------------------------------------------------
class DynamicSharedEmbedding(SharedEmbedding):
    """Creates a single embedding for a batch of data to be used by SharedEmbeddingHyperNet"""
    def __init__(self, top_hypernet: SharedEmbeddingHyperNet, input_shape):
        super().__init__(top_hypernet, 1)
        self.batch_size = input_shape[0]
        self.top_hypernet = top_hypernet
        self.build(input_shape, input_shape[1:])
    
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

# --------------------------------------------------------------------------------
#             SHARED EMBEDDING STATIC HYPERNETWORK ARCHITECTURE
# --------------------------------------------------------------------------------
#             x                                        independent embed
#             |                                            /   |   \
#             |                                         [hypernetworks] ----\
#             |                                                |   ^--------/
#             |                                                |
#             |                                                |
#             \--------------> [output network] <--------------/
# --------------------------------------------------------------------------------

# no need to subclass this as `self.weight_generator` is predefined as an `nn.Embedding` layer
class StaticSharedEmbedding(SharedEmbedding):
    def __init__(self, top_hypernet: SharedEmbeddingHyperNet, num_embeddings, embedding_dim, input_shape):
        super().__init__(top_hypernet, num_embeddings)
        self.context_vector = torch.arange(num_embeddings)
        self.top_hypernet = top_hypernet
        self.build(input_shape, embedding_dim)
    
    def create_params(self):
        self.weight_generator = nn.Sequential(nn.Embedding(self.num_embeddings, np.prod(self.embedding_dim)))
    
    def embed_and_propagate(self, raw):
        embed = self.weight_generator(self.context_vector)
        return self.top_hypernet.propagate_forward(raw, embed)
