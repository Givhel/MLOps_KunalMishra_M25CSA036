import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class CIFAR10CustomDataset(Dataset):
    """
    Custom Dataset class for CIFAR-10
    """

    def __init__(self, train=True):
        """
        Args:
            train (bool): If True, load training data, else load test data
        """
        self.train = train

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        # Load CIFAR-10 dataset internally
        self.dataset = datasets.CIFAR10(
            root="./data",
            train=self.train,
            download=True,
            transform=self.transform
        )

    def __len__(self):
        """
        Returns the total number of samples
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns one sample at index idx
        """
        image, label = self.dataset[idx]
        return image, label
