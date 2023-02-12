import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from setup import setup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from cvt.cvt import CvT
from cvt.dataset import FrameDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model_name, size, n_epochs, batch_size, output_dir, lr):
    model = CvT(size, 3, 4).to(device)
    df = pd.read_csv("dataset/ground_truth.csv")
    df = df[["filename", "class"]]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_loader = DataLoader(FrameDataset(
        train_df), batch_size=batch_size, shuffle=True)

    # Define the loss function
    loss_fun = nn.CrossEntropyLoss().to(device)

    # Choose an optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.train()
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        running_loss = 0.0
        for image, label in tqdm(train_loader, desc="Images"):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            output = model(image)
            loss = loss_fun(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Epoch %d training loss: %.3f' %
              (epoch + 1, running_loss / len(train_loader)))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/{model_name}.pkl")


def evaluate(model_name, size, batch_size, input_dir):
    model = CvT(size, 3, 4).to(device)
    model.load_state_dict(torch.load(f"{input_dir}/{model_name}.pkl"))
    df = pd.read_csv("dataset/ground_truth.csv")
    df = df[["filename", "class"]]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    test_loader = DataLoader(FrameDataset(
        test_df), batch_size=batch_size, shuffle=True)

    y_pred, y_true = [], []

    for image, label in tqdm(test_loader, desc="Images"):
        output = model(image)
        pred = np.argmax(output)
        y_pred.append(pred)
        y_true.append(label)
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch_size to train"
    )
    parser.add_argument("--output_dir", type=str,
                        default="./models", help="output directory")
    parser.add_argument(
        "--lr", type=float, default=0.02, help="learning rate"
    )
    parser.add_argument(
        "--size", type=float, default=224, help="size of the images given to train"
    )
    parser.add_argument(
        '--setup', default=None
    )
    parser.add_argument(
        '--train', default=None, help="model name"
    )
    parser.add_argument(
        '--evaluate', default=None, help="model name"
    )
    args = parser.parse_args()

    if args.setup:
        setup(args.size)

    if args.train:
        train(args.train, size=args.size, n_epochs=args.n_epochs,
              batch_size=args.batch_size, output_dir=args.output_dir, lr=args.lr)

    if args.evaluate:
        evaluate(args.evaluate)


if __name__ == '__main__':
    main()
