import torch
import numpy as np
import matplotlib.pyplot as plt


def display_traning_result(losses=None, accuracies=None):
    x = np.arange(1, len(losses)+1)
    y1 = np.array(losses)
    y2 = np.array(accuracies)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    # 损失子图
    ax1.plot(x, y1, marker="o", color="b")
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # 准确率子图
    ax2.plot(x, y2, marker="s", color="r")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_all_layers_histogram_comparison(model, bins=30):
    """
    绘制所有层权重的直方图对比
    """
    plt.figure(figsize=(15, 10))
    
    # 收集所有层的权重
    layer_weights = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            layer_weights.append(param.data.flatten())
            layer_names.append(name)
    
    # 绘制每个层的直方图
    for i, (weights, name) in enumerate(zip(layer_weights, layer_names)):
        # 使用torch.histc
        hist = torch.histc(weights, bins=bins, min=weights.min().item(), max=weights.max().item())
        bin_edges = torch.linspace(weights.min().item(), weights.max().item(), bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        plt.plot(bin_centers.numpy(), hist.numpy(), label=name, linewidth=2, alpha=0.8)
    
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Weight Distributions of All Layers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
