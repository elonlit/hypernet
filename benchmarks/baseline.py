import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from avalanche.benchmarks.classic import (
    SplitMNIST, 
    SplitCIFAR10, 
    PermutedMNIST,
    SplitCIFAR100,
    SplitTinyImageNet,
    SplitOmniglot,
    SplitCORe50
)
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive

class BaselineNetwork(nn.Module):
    """Simple baseline network consisting of two layers"""
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.features(x)
        outputs = self.classifier(features)
        return outputs

class CNNNetwork(nn.Module):
    """CNN network for image-based tasks"""
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_benchmark(benchmark_name, n_experiences):
    """Create benchmark based on name"""
    if benchmark_name == "split_mnist":
        return SplitMNIST(n_experiences=n_experiences, seed=42)
    elif benchmark_name == "split_cifar10":
        return SplitCIFAR10(n_experiences=n_experiences, seed=42)
    elif benchmark_name == "permuted_mnist":
        return PermutedMNIST(n_experiences=n_experiences, seed=42)
    elif benchmark_name == "split_cifar100":
        return SplitCIFAR100(n_experiences=n_experiences, seed=42)
    elif benchmark_name == "split_tiny_imagenet":
        return SplitTinyImageNet(n_experiences=n_experiences, seed=42)
    elif benchmark_name == "split_omniglot":
        return SplitOmniglot(n_experiences=n_experiences, seed=42)
    elif benchmark_name == "split_core50":
        return SplitCORe50(n_experiences=n_experiences, seed=42)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

def create_model(benchmark_name):
    """Create model based on benchmark"""
    if benchmark_name == "split_mnist":
        return BaselineNetwork(784, 256, 10)
    elif benchmark_name == "split_cifar10":
        return CNNNetwork(3, 10)
    elif benchmark_name == "permuted_mnist":
        return BaselineNetwork(784, 256, 10)
    elif benchmark_name == "split_cifar100":
        return CNNNetwork(3, 100)
    elif benchmark_name == "split_tiny_imagenet":
        return CNNNetwork(3, 200)
    elif benchmark_name == "split_omniglot":
        return BaselineNetwork(784, 512, 964)
    elif benchmark_name == "split_core50":
        return CNNNetwork(3, 50)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

def train_and_evaluate(benchmark_name, n_experiences, device="cuda"):
    """Main training and evaluation function"""
    benchmark = get_benchmark(benchmark_name, n_experiences)
    
    model = create_model(benchmark_name).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger(open(f"logs_baseline_{benchmark_name}.txt", "w"))]
    )
    
    strategy = Naive(
        model,
        optimizer,
        F.cross_entropy,
        train_mb_size=64,
        eval_mb_size=64,
        device=device,
        evaluator=eval_plugin
    )
    
    print(f"Starting training on {benchmark_name}")
    results = []
    for experience in benchmark.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        print(f"This experience contains {len(experience.dataset)} samples for the following classes: {experience.classes_in_this_experience}")
        strategy.train(experience)
        results.append(strategy.eval(benchmark.test_stream))
    
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    benchmarks = [
        ("split_mnist", 5),
        ("split_cifar10", 5),
        ("permuted_mnist", 10),
        ("split_cifar100", 10),
        ("split_tiny_imagenet", 10),
        ("split_omniglot", 20),
        ("split_core50", 8)
    ]
    
    for benchmark_name, n_experiences in benchmarks:
        print(f"\nRunning experiment on {benchmark_name}")
        results = train_and_evaluate(benchmark_name, n_experiences, device)
        
        print(f"\nFinal results for {benchmark_name}:")
        final_accuracy = results[-1]['Top1_Acc_Stream/eval_phase/test_stream/Task000']
        print(f"Final accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()