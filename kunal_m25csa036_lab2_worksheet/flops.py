import torch
from fvcore.nn import FlopCountAnalysis
from model import SimpleCNN


def calculate_flops():
    """
    Calculates FLOPs for the SimpleCNN model
    """
    model = SimpleCNN()
    model.eval()

    # Dummy input for CIFAR-10
    dummy_input = torch.randn(1, 3, 32, 32)

    flops = FlopCountAnalysis(model, dummy_input)

    print("Total FLOPs:", flops.total())
    print("Total FLOPs (in MFLOPs):", flops.total() / 1e6)


if __name__ == "__main__":
    calculate_flops()
