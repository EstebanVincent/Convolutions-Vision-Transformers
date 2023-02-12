from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    def __init__(self, df):
        self.filenames = df["filename"].tolist()
        self.images = []
        self.labels = df["class"].tolist()

        for filename in self.filenames:
            img = Image.open(f"dataset/imgs/{filename}")
            # Convert the image to a PyTorch tensor
            img_tensor = torch.from_numpy(
                np.array(img)).permute(2, 0, 1)

            # Normalize the tensor by dividing by 255
            img_tensor = img_tensor.float() / 255
            self.images.append(img_tensor)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])
