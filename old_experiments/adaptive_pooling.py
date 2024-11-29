import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
import torch.func
from math import ceil

from itertools import chain

class CombineOverBatchSize(nn.Module):
   def __init__(self, batch_size):
       super().__init__()
       self.batch_size = batch_size
       self.net = nn.Linear(batch_size, 1)


   def forward(self, x):
       # x of shape (B, ---)
       x = x.view(*x.shape[1:], -1) # (---, B)
       x = F.relu(self.net(x)) # (---, 1)
       x = x.view(-1, *x.shape[:-1]) # (1, ---)
       return x

class FunctionalParamVectorWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super(FunctionalParamVectorWrapper, self).__init__()
        self.module = module

    def forward(self, param_vector: torch.Tensor, x: torch.Tensor, *args, **kwargs):
        params = {}
        start = 0
        considered = self.module.only_here if isinstance(self.module, HyperNet) else self.module.named_parameters()
        for name, p in considered:
            end = start + np.prod(p.size())
            params[name] = param_vector[start:end].view(p.size())
            start = end

        if isinstance(self.module, HyperNet):
            out = torch.func.functional_call(self.module, params, (x,))
            return self.module.propagate(out, x, *args, **kwargs)
        else:
            out = torch.func.functional_call(self.module, params, (x, *args), kwargs)
            return out

class HyperNet(nn.Module):
    def __init__(self, target_network, name, device="cpu"):
        super().__init__()
        self.name = name
        self.device = device
        self.target_network = target_network
        
        if isinstance(target_network, HyperNet):
            self.total_params_target = target_network.total_only_here
        else:
            self.total_params_target = int(sum(p.numel() for p in target_network.parameters()))

        self.create_params()

        self.only_here = self.weight_generator.named_parameters()
        self.total_only_here = sum(p.numel() for _, p in self.only_here)

        self.only_here_with_comb = chain(self.weight_generator.parameters(), self.combine.parameters())
        self.total_only_here_with_comb = sum(p.numel() for p in self.only_here_with_comb)


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
        self.combine = CombineOverBatchSize(64)
        
        class WeightGenerator(nn.Module):
            def __init__(self, total_params_target, last_pool_size):
                super().__init__()
                final_channel_dim = ceil(total_params_target / (last_pool_size ** 2))

                # very tiny CNN
                self.conv1 = nn.Conv2d(1, final_channel_dim, 3, 1, 1)
                self.pool = nn.AdaptiveAvgPool2d((last_pool_size, last_pool_size))
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return x
        
        self.weight_generator = WeightGenerator(self.total_params_target, 8)

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = self.combine(x)
        x = self.weight_generator(x)
        return x

class TwoUp(HyperNet):
    def __init__(self, target_network, name):
        super().__init__(target_network, name)

    def create_params(self):
        self.combine = CombineOverBatchSize(64)

        class WeightGenerator(nn.Module):
            def __init__(self, total_params_target, last_pool_size):
                super().__init__()
                final_channel_dim = ceil(total_params_target / (last_pool_size ** 2))

                # very tiny CNN
                self.conv1 = nn.Conv2d(1, final_channel_dim, 3, 1, 1)
                self.pool = nn.AdaptiveAvgPool2d((last_pool_size, last_pool_size))
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return x
        
        self.weight_generator = WeightGenerator(self.total_params_target, 4)

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = self.combine(x)
        x = self.weight_generator(x)
        return x

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
print(f"Middle hypernetwork parameters: {one.total_only_here_with_comb}")
print(f"Top hypernetwork parameters: {top.total_only_here_with_comb}")

# Train the hypernetwork
optimizer = optim.Adam(top.only_here_with_comb, lr=1e-4, weight_decay=1e-3)

num_epochs = 10
tq = tqdm(range(num_epochs))

for ep in tq:
    for i, (x_batch, y_batch) in enumerate(list(train_loader)[:-1]):
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
    for data in list(test_loader)[:-1]:
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