# Example to demonstrate what an experience is
from avalanche.benchmarks.classic import SplitMNIST
import matplotlib.pyplot as plt
import torch

def visualize_experiences():
    # Create SplitMNIST with 5 experiences
    benchmark = SplitMNIST(n_experiences=5, seed=42)
    
    # Each experience contains different digits
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, experience in enumerate(benchmark.train_stream):
        # Print information about this experience
        print(f"\nExperience {i}:")
        print(f"Contains digits: {experience.classes_in_this_experience}")
        print(f"Number of samples: {len(experience.dataset)}")
        print(f"Task label: {experience.task_label}")
        
        # Show first image from each experience
        print(type(experience.dataset))
        img, label = experience.dataset[0]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f'Digit {label}')
        axes[i].axis('off')
    
    plt.show()

# For SplitMNIST:
visualize_experiences()