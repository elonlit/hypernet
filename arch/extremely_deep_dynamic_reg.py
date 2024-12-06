"""
This is an implementation of a deep dynamic nested hypernetwork that uses smooth weights
(exponential moving average) in the top hypernet to stabilize training.

Run using: python -m arch.EMA_hypernet
"""

from collections import defaultdict
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
from torch.nn.utils import parameters_to_vector

from krish_experiments.hypernet_lib import BaseNet, SharedEmbeddingHyperNet, DynamicSharedEmbedding
torch.manual_seed(42)
np.random.seed(42)

class Lowest(BaseNet):
    def create_params(self):
        self.weight_generator = nn.Sequential(
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(784, 100),
            nn.GELU(),
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
two = AvgPoolHyperNet(one, 4, num_backward_connections=2, device=device).to(device)
three = AvgPoolHyperNet(two, 4, num_backward_connections=2, device=device).to(device)
four = AvgPoolHyperNet(three, 4, num_backward_connections=2, device=device).to(device)

embed = LinearSharedEmbedding(four, input_shape=(64, 1, 28, 28)).to(device)

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

print(f"Embed hypernetwork parameters: {embed.num_weight_gen_params}")

num_epochs = 25
tq = tqdm(range(num_epochs))

# Train the hypernetwork
trainable_params = chain(embed.weight_generator.parameters(), four.weight_generator.parameters(),
                        [four.res_connection_vector], [three.res_connection_vector], [two.res_connection_vector],
                        [one.res_connection_vector], [base.res_connection_vector])

def print_optimized_parameters(optimizer):
    print("Parameters being optimized:")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Parameter group {i}:")
        for j, p in enumerate(param_group['params']):
            if p.requires_grad:
                print(f"  Parameter {j}:")
                print(f"    Shape: {p.shape}")
                print(f"    Data type: {p.dtype}")
                print(f"    Device: {p.device}")
                
                # Try to find the name of the parameter
                for name, param in chain(embed.named_parameters(), 
                                         two.named_parameters(), 
                                         one.named_parameters(), 
                                         base.named_parameters()):
                    if param is p:
                        print(f"    Name: {name}")
                        break
                else:
                    print("    Name: Unknown")
                
                print(f"    Requires grad: {p.requires_grad}")
                print()

#optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-3)
optimizer = optim.Adam(trainable_params, lr=1e-3)
# Call the function to print the parameters
print_optimized_parameters(optimizer)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


class HierarchicalRegularizedLoss(nn.Module):
    def __init__(self, alpha=0.01, beta=0.005, gamma=0.001, delta=0.1):
        super().__init__()
        self.alpha = alpha  # Regularization strength for the top hypernet
        self.beta = beta    # Regularization strength for the middle hypernet
        self.gamma = gamma  # Regularization strength for the base network
        self.delta = delta  # Weight for the entropy of generated weights

    def forward(self, logits, targets, embed, four, three, two, one, base):
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets)

        # L2 regularization for the top hypernet (embed)
        embed_reg = torch.norm(parameters_to_vector(embed.weight_generator.parameters()), p=2)

        # L2 regularization for the upper hypernet (four)
        four_reg = torch.norm(parameters_to_vector(four.weight_generator.parameters()), p=2)

        # L2 regularization for the middle hypernet (three)
        three_reg = torch.norm(parameters_to_vector(three), p=2)

        # L2 regularization for the middle hypernet (two)
        two_reg = torch.norm(parameters_to_vector(two), p=2)

        # L2 regularization for the lower hypernet (one)
        one_reg = torch.norm(parameters_to_vector(one), p=2)

        # L2 regularization for the base network
        base_reg = torch.norm(parameters_to_vector(base), p=2)

        # Entropy regularization for generated weights
        # generated_weights = torch.cat([p.view(-1) for p in base.parameters()])
        # weight_entropy = -torch.sum(F.softmax(generated_weights, dim=0) * F.log_softmax(generated_weights, dim=0))

        # Combine all terms
        total_loss = ce_loss + \
                     self.alpha * embed_reg + \
                     self.beta * four_reg + \
                     self.beta * three_reg + \
                     self.beta * two_reg + \
                     self.beta * one_reg + \
                     self.gamma * base_reg \
                    #  self.delta * weight_entropy

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'embed_reg': embed_reg.item(),
            'four_reg': four_reg.item(),
            'three_reg': three_reg.item(),
            'two_reg': two_reg.item(),
            'one_reg': one_reg.item(),
            'base_reg': base_reg.item(),
            # 'weight_entropy': weight_entropy.item()
        }
    
criterion = HierarchicalRegularizedLoss(alpha=0.01, beta=0.01, gamma=0.01, delta=0.01)

for ep in tq:
    epoch_losses = defaultdict(float)
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()

        logits = embed.embed_and_propagate(x_batch)
        # print(len(DynamicSharedEmbedding.get_predicted_param_list()))
        three, two, one, base = DynamicSharedEmbedding.get_predicted_param_list()
        # print(len(DynamicSharedEmbedding.get_predicted_param_list()))
        
        loss, loss_components = criterion(logits, y_batch, embed, four, three, two, one, base)
        loss.backward()

        optimizer.step()
        
        # Accumulate losses
        for k, v in loss_components.items():
            epoch_losses[k] += v

    scheduler.step()

    # Calculate average losses for the epoch
    avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
    tq.set_postfix(avg_losses)

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
