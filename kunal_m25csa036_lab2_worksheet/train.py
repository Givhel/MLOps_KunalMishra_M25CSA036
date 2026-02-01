import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb

from dataset import CIFAR10CustomDataset
from model import SimpleCNN
from utils import plot_gradient_flow, clone_model_weights, get_weight_update_norms


def train():
    # -----------------------------
    # WandB Initialization
    # -----------------------------
    wandb.init(
        project="dlops-lab2-cifar10",
        config={
            "epochs": 25,
            "batch_size": 64,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "model": "SimpleCNN",
            "dataset": "CIFAR-10"
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Dataset & DataLoader
    # -----------------------------
    train_dataset = CIFAR10CustomDataset(train=True)
    test_dataset = CIFAR10CustomDataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # -----------------------------
    # Model, Loss, Optimizer
    # -----------------------------
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(25):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        weight_update_accumulator = {}

        print(f"\nEpoch [{epoch + 1}/25]")
        progress_bar = tqdm(train_loader, desc="Training", leave=True)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Save weights BEFORE update
            prev_weights = clone_model_weights(model)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # -----------------------------
            # Accumulate Weight Updates
            # -----------------------------
            update_norms = get_weight_update_norms(model, prev_weights)
            for k, v in update_norms.items():
                weight_update_accumulator[k] = weight_update_accumulator.get(k, 0) + v

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Live progress update
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct / total:.2f}%"
            })

        # -----------------------------
        # Epoch Metrics
        # -----------------------------
        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # -----------------------------
        # Gradient Flow (ONCE PER EPOCH)
        # -----------------------------
        grad_plot = plot_gradient_flow(model.named_parameters())
        wandb.log({"Gradient Flow": wandb.Image(grad_plot)})
        grad_plot.close()

        # -----------------------------
        # Average Weight Updates
        # -----------------------------
        avg_weight_updates = {
            f"weight_update/{k}": v / len(train_loader)
            for k, v in weight_update_accumulator.items()
        }

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        # -----------------------------
        # WandB Logging
        # -----------------------------
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            **avg_weight_updates
        })

        print(
            f"Epoch [{epoch+1}/25] "
            f"Loss: {avg_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Acc: {val_acc:.2f}%"
        )

    wandb.finish()


if __name__ == "__main__":
    train()
