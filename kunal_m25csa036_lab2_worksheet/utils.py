import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_gradient_flow(named_parameters):
    """
    Plots the gradient flow through different layers in the network.
    """
    ave_grads = []
    layers = []

    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None and "bias" not in name:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())

    plt.figure(figsize=(10, 5))
    plt.plot(ave_grads, alpha=0.7)
    plt.hlines(0, 0, len(ave_grads) - 1, linewidth=1, color="k")
    plt.xticks(range(len(layers)), layers, rotation="vertical")
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.tight_layout()

    return plt


def get_weight_update_norms(model, prev_weights):
    """
    Computes weight update norms between optimizer steps.
    """
    update_norms = {}

    for name, param in model.named_parameters():
        if param.requires_grad and name in prev_weights:
            update = param.data - prev_weights[name]
            update_norms[name] = torch.norm(update).item()

    return update_norms


def clone_model_weights(model):
    """
    Stores a copy of model weights before optimizer step.
    """
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
