import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, PermutedMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive

class BaselineNetwork(nn.Module):
    """Simple baseline network that can be replaced with your hypernetwork"""
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

def get_benchmark(benchmark_name, n_experiences):
    """Create benchmark based on name"""
    if benchmark_name == "split_mnist":
        return SplitMNIST(n_experiences=n_experiences, seed=42)
    elif benchmark_name == "split_cifar10":
        return SplitCIFAR10(n_experiences=n_experiences, seed=42)
    elif benchmark_name == "permuted_mnist":
        return PermutedMNIST(n_experiences=n_experiences, seed=42)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

def create_model(benchmark_name):
    """Create model based on benchmark"""
    if benchmark_name == "split_mnist":
        return BaselineNetwork(784, 256, 10)
    elif benchmark_name == "split_cifar10":
        return BaselineNetwork(3072, 512, 10)
    elif benchmark_name == "permuted_mnist":
        return BaselineNetwork(784, 256, 10)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

def train_and_evaluate(benchmark_name, n_experiences, device="cuda"):
    """Main training and evaluation function"""
    # Create benchmark
    benchmark = get_benchmark(benchmark_name, n_experiences)
    
    # Create model
    model = create_model(benchmark_name).to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Define evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger(open(f"logs_{benchmark_name}.txt", "w"))]
    )
    
    # Create strategy
    strategy = Naive(
        model,
        optimizer,
        F.cross_entropy,
        train_mb_size=128,
        eval_mb_size=128,
        device=device,
        evaluator=eval_plugin
    )
    
    # Training loop
    print(f"Starting training on {benchmark_name}")
    results = []
    for experience in benchmark.train_stream:
        print(f"Start training on experience {experience.current_experience}")
        strategy.train(experience)
        results.append(strategy.eval(benchmark.test_stream))
    
    return results

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # List of benchmarks to evaluate
    benchmarks = [
        ("split_mnist", 5),
        ("split_cifar10", 5),
        ("permuted_mnist", 10)
    ]
    
    # Run experiments
    for benchmark_name, n_experiences in benchmarks:
        print(f"\nRunning experiment on {benchmark_name}")
        results = train_and_evaluate(benchmark_name, n_experiences, device)
        
        # Print final results
        print(f"\nFinal results for {benchmark_name}:")
        final_accuracy = results[-1]['Top1_Acc_Stream/eval_phase/test_stream/Task000']
        print(f"Final accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()