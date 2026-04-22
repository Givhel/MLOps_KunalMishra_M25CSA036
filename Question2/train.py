import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import CityDataset
import matplotlib.pyplot as plt

# ======================
# METRICS FUNCTIONS
# ======================

def compute_iou(preds, masks, num_classes=23):
    preds = torch.argmax(preds, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (masks == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            continue

        ious.append(intersection / union)

    return torch.mean(torch.tensor(ious))


def compute_dice(preds, masks, num_classes=23):
    preds = torch.argmax(preds, dim=1)
    dices = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (masks == cls)

        intersection = (pred_inds & target_inds).sum().float()
        total = pred_inds.sum() + target_inds.sum()

        if total == 0:
            continue

        dices.append((2 * intersection) / total)

    return torch.mean(torch.tensor(dices))


# ======================
# DATA LOADING
# ======================

img_paths = sorted(glob.glob("../data/CameraRGB/*.png"))
mask_paths = sorted(glob.glob("../data/CameraMask/*.png"))

train_img, test_img, train_mask, test_mask = train_test_split(
    img_paths, mask_paths, test_size=0.2, random_state=42
)

train_ds = CityDataset(train_img, train_mask)
test_ds = CityDataset(test_img, test_mask)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)

# ======================
# MODEL
# ======================

class SimpleUNet(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,num_classes,1)

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x


model = SimpleUNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ======================
# TRAINING
# ======================

losses = []
ious = []
dices = []

for epoch in range(7):
    total_loss = 0

    for imgs, masks in train_loader:
        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # compute metrics on last batch (fast approach)
    iou = compute_iou(preds, masks)
    dice = compute_dice(preds, masks)

    losses.append(total_loss)
    ious.append(iou.item())
    dices.append(dice.item())

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}")

# ======================
# TEST EVALUATION
# ======================

model.eval()
test_ious = []
test_dices = []

with torch.no_grad():
    for imgs, masks in test_loader:
        preds = model(imgs)

        iou = compute_iou(preds, masks)
        dice = compute_dice(preds, masks)

        test_ious.append(iou.item())
        test_dices.append(dice.item())

mean_iou = sum(test_ious) / len(test_ious)
mean_dice = sum(test_dices) / len(test_dices)

print("\n===== FINAL TEST RESULTS =====")
print(f"mIOU: {mean_iou:.4f}")
print(f"mDICE: {mean_dice:.4f}")

# ======================
# SAVE MODEL
# ======================

torch.save(model.state_dict(), "model.pth")

# ======================
# PLOTS
# ======================

plt.plot(losses)
plt.title("Training Loss")
plt.savefig("loss.png")
plt.clf()

plt.plot(ious)
plt.title("IoU")
plt.savefig("iou.png")
plt.clf()

plt.plot(dices)
plt.title("Dice")
plt.savefig("dice.png")
