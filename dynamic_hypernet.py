import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
# from functorch import make_functional

# class FunctionalParamVectorWrapper(nn.Module):
#     def __init__(self, module: nn.Module):
#         super(FunctionalParamVectorWrapper, self).__init__()
#         self.functional, _ = make_functional(module)
#         self.module = module

#     def to(self, device):
#         super().to(device)
#         self.module.to(device)
#         return self

#     def forward(self, param_vector: torch.Tensor, x: torch.Tensor, *args, **kwargs):
#         params = []
#         start = 0
#         considered = self.module.only_here if isinstance(self.module, HyperNet) else self.module.parameters()
#         for p in considered:
#             end = start + np.prod(p.size())
#             params.append(param_vector[start:end].view(p.size()))
#             start = end

#         if isinstance(self.module, HyperNet):
#             out = self.functional(params, x) # Parameterizes target network with param_vector and does forward pass
#             return self.module.propagate(out, x, *args, **kwargs) # Applies generated weights to its own target network
#         else:
#             out = self.functional(params, x, *args, **kwargs)
#             return out

import torch.func

class FunctionalParamVectorWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super(FunctionalParamVectorWrapper, self).__init__()
        self.module = module

    def forward(self, param_vector: torch.Tensor, x: torch.Tensor, *args, **kwargs):
        params = {}
        start = 0
        considered = self.module.only_here if isinstance(self.module, HyperNet) else self.module.parameters()
        
        for name, p in self.module.named_parameters():
            end = start + np.prod(p.size())
            params[name] = param_vector[start:end].view(p.size())
            start = end

        if isinstance(self.module, HyperNet):
            out = torch.func.functional_call(self.module, params, (x,))
            return self.module.propagate(out, x, *args, **kwargs)
        else:
            out = torch.func.functional_call(self.module, params, (x, *args), kwargs)
            return out

# class HyperNet(nn.Module):
#     def __init__(self, target_network, name, device="cpu"):
#         self.name = name
#         self.device = device
#         super().__init__()
#         self.target_network = target_network

#         if isinstance(target_network, HyperNet):
#             self.total_params_target = target_network.only_here
#         else:
#             self.total_params_target = sum(p.numel() for p in target_network.parameters())

#         self.create_params()
#         self.only_here = [param for param in self.weight_generator.parameters()]
#         self.total_params = sum(p.numel() for p in self.parameters())
#         self.update_params = FunctionalParamVectorWrapper(target_network)
    
#     def to(self, device):
#         super().to(device)
#         self.device = device
#         self.update_params.to(device)
#         if isinstance(self.target_network, HyperNet):
#             self.target_network.to(device)
#         return self
    
#     """
#     Generates weights and applies them to target network.

#     Only called by earliest hypernet in the chain.
#     """
#     def propagate_forward(self, x, *args, **kwargs):
#         out = self.forward(x)
#         return self.propagate(out, x, *args, **kwargs)

#     """
#     Apply generated weights to target network.
#     """
#     def propagate(self, out, x, *args, **kwargs):
#         # This is a recursive call to parameterize target network,
#         # call forward pass to generate weights, and apply them to target network,
#         # and so on until the base network.
#         return self.update_params(out.view(-1), x, *args, **kwargs)

#     def create_params(self):
#         raise NotImplementedError("Subclasses implement this method to initialize the weight generator.")

#     def forward(self, x):
#         raise NotImplementedError("Subclasses implement this method to generate weights for target network.")

class HyperNet(nn.Module):
    def __init__(self, target_network, name, device="cpu"):
        super().__init__()
        self.name = name
        self.device = device
        self.target_network = target_network

        # Fix the parameter counting
        if isinstance(target_network, HyperNet):
            self.total_params_target = sum(p.numel() for p in target_network.only_here)
        else:
            self.total_params_target = sum(p.numel() for p in target_network.parameters())

        self.create_params()
        self.only_here = [param for param in self.weight_generator.parameters()]
        self.total_params = sum(p.numel() for p in self.parameters())
        self.update_params = FunctionalParamVectorWrapper(target_network)
    
    def to(self, device):
        super().to(device)
        self.device = device
        self.update_params.to(device)
        if isinstance(self.target_network, HyperNet):
            self.target_network.to(device)
        return self
    
    def propagate_forward(self, x, *args, **kwargs):
        out = self.forward(x)
        return self.propagate(out, x, *args, **kwargs)

    def propagate(self, out, x, *args, **kwargs):
        return self.update_params(out.view(-1), x, *args, **kwargs)

    def create_params(self):
        raise NotImplementedError("Subclasses implement this method to initialize the weight generator.")

    def forward(self, x):
        raise NotImplementedError("Subclasses implement this method to generate weights for target network.")
############################################ TESTING ############################################

class Lowest(nn.Module):
    def __init__(self, name="L"):
        self.name = name
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(784, 10),
        )

    def forward(self, x):
        out = self.net(x)
        return out

class OneUp(HyperNet):
    def __init__(self, target_network, name):
        super().__init__(target_network, name)

    def create_params(self):
        self.weight_generator = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, int(self.total_params_target))
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.weight_generator(x)

class TwoUp(HyperNet):
    def __init__(self, target_network, name):
        super().__init__(target_network, name)

    def create_params(self):
        self.weight_generator = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, int(self.total_params_target))
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.weight_generator(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
base = Lowest().to(device)
one = OneUp(base, name="One-up hypernetwork").to(device)
top = TwoUp(one, name="Two-up hypernetwork").to(device)



# Load MNIST dataset
morph = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root="data/", train=True, transform=morph, download=True)
test_set = torchvision.datasets.MNIST(root="data/", train=False, transform=morph, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

print(f"Base network parameters: {sum(p.numel() for p in base.parameters())}")
print(f"Top hypernetwork parameters: {sum(p.numel() for p in top.only_here)}")

# Train the hypernetwork
optimizer = optim.Adam(top.parameters(), lr=1e-4, weight_decay=1e-3)

num_epochs = 10
tq = tqdm(range(num_epochs))

for ep in tq:
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()

        logits = top.propagate_forward(x_batch)
        
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        optimizer.step()

    tq.set_postfix(loss=loss.item())

# Evaluate the hypernetwork
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = top.propagate_forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

hypernet_acc = correct / total
print(f"Nested hypernet accuracy: {hypernet_acc:.4f}")

# Train the base network for comparison
optimizer = optim.Adam(base.parameters(), lr=1e-4, weight_decay=1e-3)

tq = tqdm(range(num_epochs))

for ep in tq:
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()

        logits = base(x_batch)
        
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        optimizer.step()

    tq.set_postfix(loss=loss.item())

# Evaluate the base network
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = base(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

base_acc = correct / total
print(f"Base network accuracy: {base_acc:.4f}")

print(f"Improvement: {(hypernet_acc - base_acc) * 100:.2f}%")