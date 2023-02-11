import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cvt.cvt import CvT
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

model = CvT(128, 3, 4)

df = pd.read_csv("dataset/ground_truth.csv")
df = df[["filename", "class"]]

train_df, test_df = train_test_split(df, test_size=0.2)


class FrameDataset(Dataset):
    def __init__(self, df):
        self.filenames = df["filename"].tolist()
        self.images = []
        self.labels = df["class"].tolist()

        for filename in tqdm(self.filenames):
            img = Image.open(f"dataset/imgs/{filename}")
            # Convert the image to a PyTorch tensor
            img_tensor = torch.from_numpy(
                np.array(img)).permute(2, 0, 1).unsqueeze(0)

            # Normalize the tensor by dividing by 255
            img_tensor = img_tensor.float() / 255
            self.images.append(img_tensor)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])


train_loader = DataLoader(FrameDataset(train_df), batch_size=1, shuffle=True)

# Define the loss function
loss_fun = nn.CrossEntropyLoss()

# Choose an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.02)

# Train the model
model.train()
for epoch in tqdm(range(1), desc="Epochs"):
    running_loss = 0.0
    for image, label in tqdm(train_loader, desc="Images"):
        optimizer.zero_grad()

        output = model(torch.squeeze(image, 0))
        loss = loss_fun(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Epoch %d training loss: %.3f' %
          (epoch + 1, running_loss / len(train_loader)))
