"""
This file tests how deep we can go using the dynamic hypernet approach on MNIST.
Using n = 4 hypernets.

Run using `python -m arch.extremely_deep_dynamic` from the root directory.
"""

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

from krish_experiments.hypernet_lib import BaseNet, SharedEmbeddingHyperNet, DynamicSharedEmbedding
torch.manual_seed(42)
np.random.seed(42)

def init_master_hypernetwork(hypernet):
    if hasattr(hypernet, 'weight_generator') and isinstance(hypernet, AvgPoolHyperNet) and isinstance(hypernet.weight_generator.conv1, nn.Conv2d):
        kernel_size = hypernet.weight_generator.conv1.kernel_size
        n_k = kernel_size[0] * kernel_size[1]
        c_in = hypernet.weight_generator.conv1.in_channels
        pool_size = hypernet.pool_size
        
        std = torch.sqrt(2/(n_k * c_in * pool_size**2))
        nn.init.normal_(hypernet.weight_generator.conv1.weight, 0, std)
        if hypernet.weight_generator.conv1.bias is not None:
            nn.init.zeros_(hypernet.weight_generator.conv1.bias)

def init_embedding_network(embed_net):
    if isinstance(embed_net, DynamicSharedEmbedding) and isinstance(embed_net.weight_generator.linear, nn.Linear):
        batch_size = embed_net.weight_generator.linear.in_features
        nn.init.normal_(embed_net.weight_generator.linear.weight, 0, np.sqrt(2/batch_size))
        if embed_net.weight_generator.linear.bias is not None:
            nn.init.zeros_(embed_net.weight_generator.linear.bias)

def init_residual_connections(hypernet, pool_size):
    if hasattr(hypernet, 'res_connection_vector'):
        k = len(hypernet.res_connection_vector)
        std = np.sqrt(1/(k * pool_size**2))
        nn.init.normal_(hypernet.res_connection_vector, 0, std)

class Lowest(BaseNet):
    def create_params(self):
        self.weight_generator = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

class AvgPoolHyperNet(SharedEmbeddingHyperNet):
    def __init__(self, top_hypernet, pool_size, num_backward_connections=0, connection_type="avg", device="cpu"):
        super().__init__(top_hypernet, num_backward_connections, connection_type=connection_type, device=device)
        self.pool_size = pool_size

    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self, num_params_to_estimate, pool_size):
                super().__init__()
                final_channel_dim = ceil(num_params_to_estimate / (pool_size ** 2))

                # very tiny CNN
                self.conv1 = nn.Conv2d(1, final_channel_dim, 3, 1, 1)
                self.batch_norm = nn.BatchNorm2d(final_channel_dim)
                self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
                            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.batch_norm(x)
                x = self.pool(x)
                return x
        
        self.weight_generator = WeightGenerator(self.num_params_to_estimate, self.pool_size)

class MaxPoolHyperNet(SharedEmbeddingHyperNet):
    def __init__(self, top_hypernet, pool_size, num_backward_connections=0, connection_type="max", device="cpu"):
        super().__init__(top_hypernet, num_backward_connections, connection_type=connection_type, device=device)
        self.pool_size = pool_size

    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self, num_params_to_estimate, pool_size):
                super().__init__()
                final_channel_dim = ceil(num_params_to_estimate / (pool_size ** 2))

                # very tiny CNN
                self.conv1 = nn.Conv2d(1, final_channel_dim, 3, 1, 1)
                self.pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                return x
        
        self.weight_generator = WeightGenerator(self.num_params_to_estimate, self.pool_size)

class AttnSharedEmbedding(DynamicSharedEmbedding):
    def __init__(self, top_hypernet, batch_size, num_heads, dropout_rate=0.1):
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        super().__init__(top_hypernet, batch_size)

    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self, batch_size, num_heads, dropout_rate=0.1):
                super().__init__()
                self.attn = nn.MultiheadAttention(embed_dim=batch_size, num_heads=num_heads, dropout=dropout_rate)

            def forward(self, padded, diff):
                bef_attn_shape = padded.shape

                padded = padded.view(1, np.prod(padded.shape[1:]), padded.shape[0])
                padded = F.relu(self.attn(padded, padded, padded)[0])

                padded = padded.view(*bef_attn_shape)

                padded = padded[:padded.shape[0]-diff, ...]
                padded = padded.mean(dim=0, keepdim=True)

                return padded

        self.weight_generator = WeightGenerator(self.batch_size, self.num_heads, self.dropout_rate)

class LinearSharedEmbedding(DynamicSharedEmbedding):
    def create_params(self):
        class WeightGenerator(nn.Module):
            def __init__(self, batch_size):
                super().__init__()
                self.linear = nn.Linear(batch_size, 1)

            def forward(self, padded, diff):
                padded = padded.transpose(0, -1)
                embed = F.relu(self.linear(padded))
                embed = embed.transpose(0, -1)
                embed = embed / (padded.shape[0] - diff)
                # ^ this line allows other batch sizes to run, loss is artificially high, accuracy is quite good
                # although in some runs, accuracy is better without this line, so just test it out

                return embed

        self.weight_generator = WeightGenerator(self.batch_size)

device = torch.device("cpu")
print(f"Device: {device}")

# we can set num_backward_connection=2 because in the library,
# we only connect min(2, # of hypernets above) - so in `OneUp`
# we only connection `top`'s parameters and in `TwoUp` we 
# don't connect any parameters (since no hypernet above)
base = Lowest(num_backward_connections=4, device=device).to(device)

one = AvgPoolHyperNet(base, 4, num_backward_connections=3, device=device).to(device)
init_residual_connections(one, pool_size=4)

two = AvgPoolHyperNet(one, 4, num_backward_connections=2, device=device).to(device)
init_residual_connections(two, pool_size=4)

three = AvgPoolHyperNet(two, 4, num_backward_connections=2, device=device).to(device)
init_residual_connections(three, pool_size=4)

four = AvgPoolHyperNet(three, 4, num_backward_connections=2, device=device).to(device)
four.apply(init_master_hypernetwork)

init_residual_connections(four, pool_size=4)

embed = LinearSharedEmbedding(four, input_shape=(64, 1, 28, 28)).to(device)
embed.apply(init_embedding_network)

morph = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root="data/", train=True, transform=morph, download=True)
test_set = torchvision.datasets.MNIST(root="data/", train=False, transform=morph, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

print(f"Base network parameters: {base.num_weight_gen_params}")
print(f"One hypernetwork parameters: {one.num_weight_gen_params}")
print(f"Two hypernetwork parameters: {two.num_weight_gen_params}")
print(f"Three hypernetwork parameters: {three.num_weight_gen_params}")
print(f"Four hypernetwork parameters: {four.num_weight_gen_params}")
print(f"Embedding network parameters: {embed.num_weight_gen_params}")

num_epochs = 25
tq = tqdm(range(num_epochs))

# Train the hypernetwork
trainable_params = chain(embed.weight_generator.parameters(), four.weight_generator.parameters(),
                        [four.res_connection_vector], [three.res_connection_vector], [two.res_connection_vector],
                        [one.res_connection_vector], [base.res_connection_vector])

optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-3)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for ep in tq:
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()

        logits = embed.embed_and_propagate(x_batch)
        
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        optimizer.step()

    # scheduler.step()
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

base = nn.Sequential(
    nn.Flatten(),
    nn.ReLU(),
    nn.Linear(784, 10),
).to(device)

# Train the base network for comparison
optimizer = optim.AdamW(base.parameters(), lr=1e-3, weight_decay=1e-3)

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
