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

from hypernet_lib import DynamicHyperNet, DynamicEmbedding

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

class OneUp(DynamicHyperNet):
    def __init__(self, target_network):
        super().__init__(target_network)

    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self, num_params_to_estimate, last_pool_size):
                super().__init__()
                final_channel_dim = ceil(num_params_to_estimate / (last_pool_size ** 2))

                # very tiny CNN
                self.conv1 = nn.Conv2d(1, final_channel_dim, 3, 1, 1)
                self.pool = nn.AdaptiveAvgPool2d((last_pool_size, last_pool_size))
            
            def forward(self, x):
                x = x.view(x.size(0), 1, 28, 28)
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return x
        
        self.weight_generator = WeightGenerator(self.num_params_to_estimate, 8)

class TwoUp(DynamicHyperNet):
    def __init__(self, target_network):
        super().__init__(target_network)

    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self, num_params_to_estimate, last_pool_size):
                super().__init__()
                final_channel_dim = ceil(num_params_to_estimate / (last_pool_size ** 2))

                # very tiny CNN
                self.conv1 = nn.Conv2d(1, final_channel_dim, 3, 1, 1)
                self.pool = nn.AdaptiveAvgPool2d((last_pool_size, last_pool_size))
            
            def forward(self, x):
                x = x.view(x.size(0), 1, 28, 28)
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return x
        
        self.weight_generator = WeightGenerator(self.num_params_to_estimate, 4)

class TopEmbedding(DynamicEmbedding):
    def __init__(self, top_hypernet, batch_size, num_heads, dropout=0.1):
        super().__init__(top_hypernet, batch_size, num_heads, dropout)

    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, dropout=0.1)

            def forward(self, padded, diff):
                bef_attn_shape = padded.shape

                padded = padded.view(1, np.prod(padded.shape[1:]), padded.shape[0])
                padded = F.relu(self.attn(padded, padded, padded)[0])

                padded = padded.view(*bef_attn_shape)

                padded = padded[:padded.shape[0]-diff, ...]
                padded = padded.mean(dim=0, keepdim=True)

                return padded

        self.weight_generator = WeightGenerator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

base = Lowest().to(device)
one = OneUp(base).to(device)
top = TwoUp(base).to(device)
embed = TopEmbedding(top, 64, 4).to(device)

morph = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root="data/", train=True, transform=morph, download=True)
test_set = torchvision.datasets.MNIST(root="data/", train=False, transform=morph, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

print(f"Base network parameters: {sum(p.numel() for p in base.parameters())}")
print(f"Middle hypernetwork parameters: {one.num_weight_gen_params}")
print(f"Top hypernetwork parameters: {top.num_weight_gen_params}")
print(f"Embed hypernetwork parameters: {embed.num_weight_gen_params}")

# Train the hypernetwork
trainable_params = chain(embed.weight_generator.parameters(), top.weight_generator.parameters())
optimizer = optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-3)

num_epochs = 25
tq = tqdm(range(num_epochs))

for ep in tq:
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()

        logits = embed.embed_and_propagate(x_batch)
        
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

        outputs = embed.embed_and_propagate(images)
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