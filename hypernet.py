import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
from functorch import make_functional

class FunctionalParamVectorWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super(FunctionalParamVectorWrapper, self).__init__()
        self.functional, _ = make_functional(module)
        self.module = module

    def forward(self, param_vector: torch.Tensor, *args, **kwargs):
        params = []
        start = 0
        considered = self.module.only_here if isinstance(self.module, HyperNet) else self.module.parameters()
        for p in considered:
            end = start + np.prod(p.size())
            params.append(param_vector[start:end].view(p.size()))
            start = end

        if isinstance(self.module, HyperNet):
            out = self.functional(params) # Parameterizes target network with param_vector and does forward pass
            return self.module.propagate(out, *args, **kwargs) # Applies generated weights to its own target network
        else:
            out = self.functional(params, *args, **kwargs)
            return out

class HyperNet(nn.Module):
    def __init__(self, target_network, num_embeddings, embedding_dim, name):
        self.name = name
        super().__init__()
        considered = target_network.only_here if isinstance(target_network, HyperNet) else target_network.parameters()
        self.total_params = sum(p.numel() for p in considered)

        self.weight_chunk_dim = np.ceil(self.total_params / num_embeddings).astype(int)
        self.context = torch.arange(num_embeddings, dtype=torch.long)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim 

        self.create_params()
        self.only_here = [param for param in self.parameters()]

        self.update_params = FunctionalParamVectorWrapper(target_network)
        
        self.target_network = target_network
    
    """
    Generates weights and applies them to target network.

    Only called by earliest hypernet in the chain.
    """
    def propagate_forward(self, *args, **kwargs):
        out = self.forward() # (num_embeddings, weight_chunk_dim)
        return self.propagate(out, *args, **kwargs)

    """
    Apply generated weights to target network.
    """
    def propagate(self, out, *args, **kwargs):
        return self.update_params(out.view(-1), *args, **kwargs)


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
        out = self.net(x).view(x.size(0), -1)
        return out

base = Lowest()

class OneUp(HyperNet):
    def __init__(self, target_network, num_embeddings, embedding_dim, name):
        super().__init__(target_network, num_embeddings, embedding_dim, name)

    def create_params(self):
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.weight_generator = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.weight_chunk_dim)
        )

    def forward(self):
        return self.weight_generator(self.embedding(self.context))

middle = OneUp(base, 128, 32, name="M")

class TwoUp(HyperNet):
    def __init__(self, target_network, num_embeddings, embedding_dim, name):
        super().__init__(target_network, num_embeddings, embedding_dim, name)

    def create_params(self):
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.weight_generator = nn.Sequential(
            nn.Linear(self.embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, self.weight_chunk_dim)
        )
    
    def forward(self):
        return self.weight_generator(self.embedding(self.context))

top = TwoUp(middle, 128, 32, name="T")

morph = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root="data/", train=True, transform=morph, download=True)
test_set = torchvision.datasets.MNIST(root="data/", train=False, transform=morph, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

print(sum(p.numel() for p in base.parameters()))
print(sum(p.numel() for p in top.only_here))

optimizer = optim.Adam(top.only_here, lr=1e-4, weight_decay=1e-3)

tq = tqdm(range(10))

for ep in tq:
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch
        y_batch = y_batch

        optimizer.zero_grad()

        logits = top.propagate_forward(x_batch)
        
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        optimizer.step()

    tq.set_postfix(loss=loss.item())

# Print learned embeddings of middle
print(middle.embedding(middle.context))

# get the accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = top.propagate_forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

hypernet_acc = correct / total
print(correct / total)

# Train just lowest
optimizer = optim.Adam(base.parameters(), lr=1e-4, weight_decay=1e-3)

tq = tqdm(range(10))

for ep in tq:
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch
        y_batch = y_batch

        optimizer.zero_grad()

        logits = base(x_batch)
        
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()

        optimizer.step()

    tq.set_postfix(loss=loss.item())

# get the accuracy
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
print(correct / total)

# Plot params, accuracies, and networks using seaborn
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

networks = ["Lowest", "Middle", "Top"]
num_params = [sum(p.numel() for p in net.parameters()) for net in [base, top]]
accuracies = [base_acc, hypernet_acc]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.barplot(x=networks, y=num_params, palette='viridis', ax=ax[0])

ax[0].set_title('Number of Parameters in Different Networks')
ax[0].set_ylabel('Number of Parameters')
ax[0].set_ylim(0, max(num_params) * 1.1)  # Set y-axis to start at zero and go slightly above the highest bar

for i, val in enumerate(num_params):
    ax[0].text(i, val, f'{val:,}', ha='center', va='bottom')  # Use comma as thousands separator

sns.barplot(x=["Lowest", "HyperNet"], y=accuracies, palette='viridis', ax=ax[1])

ax[1].set_title('Accuracy of Different Networks')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_ylim(0, 1.1)  # Set y-axis to start at zero and go slightly above the highest bar

for i, val in enumerate(accuracies):
    ax[1].text(i, val, f'{val:.2f}', ha='center', va='bottom')  # Use comma as thousands separator

plt.tight_layout()
plt.show()
# Save with high DPI
plt.savefig('mmnist.png', dpi=300)
