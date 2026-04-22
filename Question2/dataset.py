from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class CityDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = img.resize((128, 96))
        img = np.array(img).astype(np.float32) / 255.0

        mask = Image.open(self.mask_paths[idx]).convert("RGB")
        mask = mask.resize((128, 96))
        mask = np.array(mask)

        mask = mask[:, :, 0]   # take 1 channel
        mask = mask // 11      # reduce to 0–22 classes

        img = torch.tensor(img).permute(2,0,1)
        mask = torch.tensor(mask).long()

        return img, mask
