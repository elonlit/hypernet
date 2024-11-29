import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F
from avalanche.benchmarks.classic import (
    SplitMNIST, 
    SplitCIFAR10, 
    PermutedMNIST,
    SplitCIFAR100,
    SplitTinyImageNet,
    SplitOmniglot,
    CORe50
)
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive
from avalanche.training.templates import SupervisedTemplate

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

class HyperNet(nn.Module):
    def __init__(self, target_network, name, input_size, device="cpu"):
        super().__init__()
        self.name = name
        self.device = device
        self.target_network = target_network
        self.input_size = input_size

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
    
    def propagate(self, out, x, *args, **kwargs):
        return self.update_params(out.view(-1), x, *args, **kwargs)

    def propagate_forward(self, x, *args, **kwargs):
        x = x.to(self.device)
        out = self.forward(x)
        return self.propagate(out, x, *args, **kwargs)

    def create_params(self):
        raise NotImplementedError("Subclasses implement this method to initialize the weight generator.")

    def forward(self, x):
        raise NotImplementedError("Subclasses implement this method to generate weights for target network.")

class Lowest(nn.Module):
    def __init__(self, input_size, num_classes, name="L"):
        super().__init__()
        self.name = name
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(input_size, num_classes),
        )

    def forward(self, x):
        out = self.net(x)
        return out

class CNNNetwork(nn.Module):
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

class OneUp(HyperNet):
    def __init__(self, target_network, name, input_size):
        super().__init__(target_network, name, input_size)

    def create_params(self):
        self.weight_generator = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, int(self.total_params_target))
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.weight_generator(x)

class TwoUp(HyperNet):
    def __init__(self, target_network, name, input_size):
        super().__init__(target_network, name, input_size)

    def create_params(self):
        self.weight_generator = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, int(self.total_params_target))
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.weight_generator(x)

def get_benchmark(benchmark_name, n_experiences):
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
        return CORe50(scenario="nc", run=1, n_experiences=n_experiences)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

def create_model(benchmark_name):
    if benchmark_name == "split_mnist":
        base = Lowest(784, 10)
        one = OneUp(base, name="One-up hypernetwork", input_size=784)
        return TwoUp(one, name="Two-up hypernetwork", input_size=784)
    elif benchmark_name == "split_cifar10":
        base = CNNNetwork(3, 10)
        one = OneUp(base, name="One-up hypernetwork", input_size=3072)
        return TwoUp(one, name="Two-up hypernetwork", input_size=3072)
    elif benchmark_name == "permuted_mnist":
        base = Lowest(784, 10)
        one = OneUp(base, name="One-up hypernetwork", input_size=784)
        return TwoUp(one, name="Two-up hypernetwork", input_size=784)
    elif benchmark_name == "split_cifar100":
        base = CNNNetwork(3, 100)
        one = OneUp(base, name="One-up hypernetwork", input_size=3072)
        return TwoUp(one, name="Two-up hypernetwork", input_size=3072)
    elif benchmark_name == "split_tiny_imagenet":
        base = CNNNetwork(3, 200)
        one = OneUp(base, name="One-up hypernetwork", input_size=12288)
        return TwoUp(one, name="Two-up hypernetwork", input_size=12288)
    elif benchmark_name == "split_omniglot":
        base = Lowest(784, 964)
        one = OneUp(base, name="One-up hypernetwork", input_size=784)
        return TwoUp(one, name="Two-up hypernetwork", input_size=784)
    elif benchmark_name == "split_core50":
        base = CNNNetwork(3, 50)
        one = OneUp(base, name="One-up hypernetwork", input_size=49152)
        return TwoUp(one, name="Two-up hypernetwork", input_size=49152)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

class HyperNetStrategy(SupervisedTemplate):
    def __init__(self, model, optimizer, criterion=F.cross_entropy,
                train_mb_size=1, train_epochs=1, eval_mb_size=None,
                device='cpu', plugins=None, evaluator=None,
                eval_every=-1):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every
        )
        self.criterion_fn = criterion

    def forward(self):
        # Instead of modifying mb_x, use it directly
        x = self.mb_x.to(self.device)
        # Get the output using propagate_forward
        self.mb_output = self.model.propagate_forward(x)
        return self.mb_output

    def criterion(self):
        # Use mb_y directly without modifying it
        y = self.mb_y.to(self.device)
        if self.mb_output is None:
            self.mb_output = self.forward()
        return self.criterion_fn(self.mb_output, y)

def train_and_evaluate(benchmark_name, n_experiences, device="cuda"):
    benchmark = get_benchmark(benchmark_name, n_experiences)
    
    model = create_model(benchmark_name).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create evaluation plugin
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[interactive_logger, TextLogger(open(f"logs_hypernet_{benchmark_name}.txt", "w"))]
    )
    
    # Create the strategy
    strategy = HyperNetStrategy(
        model=model,
        optimizer=optimizer,
        criterion=F.cross_entropy,
        train_mb_size=64,
        train_epochs=1,
        device=device,
        evaluator=eval_plugin
    )

    print(f"Starting training on {benchmark_name}")
    results = []
    try:
        for experience in benchmark.train_stream:
            print(f"Start training on experience {experience.current_experience}")
            print(f"This experience contains {len(experience.dataset)} samples")
            print(f"Classes: {experience.classes_in_this_experience}")
            
            strategy.train(experience)
            results.append(strategy.eval(benchmark.test_stream))
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e
    
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