import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import os
import numpy as np

from cvt.cvt import CvT
from cvt.dataset import FrameDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model_name, size, n_epochs, batch_size, output_dir, lr):
    '''Train the model and save the model and the loss and accuracy DataFrame to disk.'''
    model = CvT(size, 3, 4).to(device)
    df = pd.read_csv('dataset/ground_truth.csv')
    df = df[['filename', 'class']]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_loader = DataLoader(FrameDataset(
        train_df), batch_size=batch_size, shuffle=True)

    # Define the loss function
    loss_fun = nn.CrossEntropyLoss().to(device)

    # Choose an optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize a DataFrame to store loss and accuracy
    loss_acc = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])

    # Train the model
    model.train()
    for epoch in tqdm(range(n_epochs), desc='Epochs', position=0, leave=False):
        running_loss = 0.0
        running_acc = 0.0
        for image, label in tqdm(train_loader, desc='Images', position=0, leave=False):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            output = model(image)
            loss = loss_fun(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += (output.argmax(1) == label).float().mean().item()

        # Calculate average loss and accuracy
        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)     

        print('Epoch %d training loss: %.3f accuracy: %.3f' %
              (epoch + 1, avg_loss, avg_acc))
        # Store loss and accuracy in DataFrame
        loss_acc = loss_acc.append({'epoch': epoch + 1, 'loss': avg_loss, 'accuracy': avg_acc}, ignore_index=True)  
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{output_dir}/{model_name}.pkl')
    loss_acc.to_csv(f'{output_dir}/{model_name}_loss_acc.csv', index=False)


def evaluate(model_name, size, batch_size, input_dir):
    '''Evaluate the model on the test set and save the prediction to disk.'''
    model = CvT(size, 3, 4).to(device)
    model.load_state_dict(torch.load(f'{input_dir}/{model_name}.pkl'))
    df = pd.read_csv('dataset/ground_truth.csv')
    df = df[['filename', 'class']]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    test_loader = DataLoader(FrameDataset(
        test_df), batch_size=batch_size, shuffle=False)

    y_pred, y_true = [], []

    for image, label in tqdm(test_loader, desc='Images'):
        image = image.to(device)
        #label = label.to(device)
        output = model(image)
        y_pred.extend(output.argmax(1).tolist())
        y_true.extend(label)
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy:', accuracy)

    test_df['prediction'] = y_pred
    test_df.to_csv(f'{input_dir}/{model_name}_prediction.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument(
        '--model_name', type=str, default='cvt', help='model name'
    )
    parser.add_argument(
        '--n_epochs', type=int, default=50, help='number of epochs to train'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch_size to train'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./models', help='output directory'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='learning rate'
    )
    parser.add_argument(
        '--size', type=int, default=128, help='size of the images given to train'
    )
    parser.add_argument(
        '--setup', action='store_true', help='setup the dataset'
    )
    parser.add_argument(
        '--train', action='store_true', help='train the model'
    )
    parser.add_argument(
        '--evaluate', action='store_true', help='evaluate the model'
    )
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(42)

    # Setup the dataset
    if args.setup:
        from setup import setup
        setup(args.size)

    # Train the model
    if args.train:
        train(args.model_name, size=args.size, n_epochs=args.n_epochs,
              batch_size=args.batch_size, output_dir=args.output_dir, lr=args.lr)

    # Evaluate the model
    if args.evaluate:
        evaluate(args.model_name, args.size, args.batch_size, args.output_dir)


if __name__ == '__main__':
    main()
