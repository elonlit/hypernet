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

class FunctionalParamVectorWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super(FunctionalParamVectorWrapper, self).__init__()
        self.module = module

    def forward(self, param_vector: torch.Tensor, raw: torch.Tensor, *args, **kwargs):
        params = {}
        start = 0

        if isinstance(self.module, HyperNet):
            considered = self.module.weight_generator.named_parameters()
        else:
            considered = self.module.named_parameters()
        
        for name, p in list(considered):
            end = start + np.prod(p.size())
            params[name] = param_vector[start:end].view(p.size())
            start = end

        if isinstance(self.module, HyperNet):
            out = torch.func.functional_call(self.module.weight_generator, params, (raw,))
            return self.module.propagate(out, raw, *args, **kwargs)
        else:
            out = torch.func.functional_call(self.module, params, (raw, *args), kwargs)
            return out

class HyperNet(nn.Module):
    def __init__(self, target_network, device="cpu"):
        super().__init__()
        self.device = device
        
        if isinstance(target_network, HyperNet):
            self.num_params_to_estimate = target_network.num_weight_gen_params
        else:
            self.num_params_to_estimate = int(sum(p.numel() for p in target_network.parameters()))

        self.create_params()

        self.num_weight_gen_params = sum(p.numel() for p in self.weight_generator.parameters())
        self.target_param_updater = FunctionalParamVectorWrapper(target_network)
    
    def propagate_forward(self, raw, *args, **kwargs):
        out = self.weight_generator.forward(raw)
        return self.propagate(out, raw, *args, **kwargs)
    
    def propagate(self, out, raw, *args, **kwargs):
        return self.target_param_updater(out.view(-1), raw, *args, **kwargs)

    def create_params(self):
        raise NotImplementedError("Subclasses implement this method to initialize the weight generator.")

    def forward(self, x):
        raise NotImplementedError("Subclasses implement this method to generate weights for target network.")

############################################ TESTING ############################################

class Lowest(nn.Module):
    def __init__(self):
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
    def __init__(self, target_network):
        super().__init__(target_network)

    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self, params_to_estimate):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(784, 64),
                    nn.ReLU(),
                    nn.Linear(64, params_to_estimate),
                )
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.net(x)
        
        self.weight_generator = WeightGenerator(self.num_params_to_estimate)

class TwoUp(HyperNet):
    def __init__(self, target_network):
        super().__init__(target_network)

    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self, params_to_estimate):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(784, 64),
                    nn.ReLU(),
                    nn.Linear(64, params_to_estimate),
                )

            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.net(x)

        self.weight_generator = WeightGenerator(self.num_params_to_estimate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
base = Lowest().to(device)
one = OneUp(base).to(device)
top = TwoUp(one).to(device)

# Load MNIST dataset
morph = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root="data/", train=True, transform=morph, download=True)
test_set = torchvision.datasets.MNIST(root="data/", train=False, transform=morph, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

print(f"Base network parameters: {sum(p.numel() for p in base.parameters())}")
print(f"Middle hypernetwork parameters: {one.num_weight_gen_params}")
print(f"Top hypernetwork parameters: {top.num_weight_gen_params}")
# print(f"Embed hypernetwork parameters: {embed.num_weight_gen_params}")

# Train the hypernetwork
optimizer = optim.Adam(chain(top.weight_generator.parameters()), lr=1e-4, weight_decay=1e-3)

num_epochs = 25
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