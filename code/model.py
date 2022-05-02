from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torchinfo import summary
import pandas as pd
from collections import Counter
import time
import sys
import torchaudio
from pathlib import Path
import os


target_num_samples = 10000
birds = ['arcter', 'bcnher']


def load_audio_files(path: str, label: str):
    dataset = []
    walker = sorted(str(p) for p in Path(path).glob(f'*.ogg'))

    for i, file_path in enumerate(walker):
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext(filename)

        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)

        #Reduce first dimension to 1
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        length_waveform = waveform.shape[1]

        #Reduce second dimension to target_num_samples

        if length_waveform > target_num_samples:
            waveform = waveform[:, :target_num_samples]

        elif length_waveform < target_num_samples:
            num_missing_samples = target_num_samples - length_waveform
            last_dim_padding = (0, num_missing_samples)
            waveform = torch.nn.functional.pad(waveform, last_dim_padding)

        dataset.append([waveform, sample_rate, label, speaker])

    return dataset

class AudioDataSet(Dataset):
    def __init__(self, folder, labels):
        self.classes = labels

        self.class_to_idx = { i : labels[i] for i in range(0, len(labels)) }

        self.class_to_idx_rev = {labels[i]: i for i in range(0, len(labels))}

        self.data = []
        for name in labels:
            #audios : [ [[data]] ; sample_rate ; label ; filename]
            audios = load_audio_files(f'./{folder}/{name}', name)

            if len(audios) == 0:
                print("CHECK FILE NAMES AND DATA!!")
                sys.exit()

            self.data.extend(audios)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data[0], self.class_to_idx_rev[data[2]]

if __name__ == '__main__':
    epochs = 10
    batch_size = 15
    num_workers = 2
    learning_rate = 0.001
    train_ratio = 0.8
    stop_over = 90 # percent
    data_path = './birdclef-2022/train_audio' #looking in subfolder train

    bestAcc = 0
    bestEpoch = 0

    dataset = AudioDataSet(
        folder=data_path,
        labels = birds
    )

    print(dataset)
    print("\n {} Class category and index of the images: {}\n".format(len(dataset.classes), dataset.class_to_idx))

    # cost function used to determine best parameters
    cost = torch.nn.CrossEntropyLoss()

    class CNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

            self.conv2 = nn.Sequential(
                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

            self.conv3 = nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

            self.conv4 = nn.Sequential(
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

            self.flatten = nn.Flatten()
            #todo: number of in_features to nn.Linear has to be changed with every change of target_num_samples
            #todo: figure out number of input features by program itself .. ?
            self.linear = nn.Linear(80128, len(dataset.classes))
            self.softmax = nn.Softmax(dim=1)

        def forward(self, input_data):
            x = self.conv1(input_data)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.flatten(x)
            logits = self.linear(x)
            predictions = self.softmax(logits)
            return predictions



    def train(dataloader, model, cost, optimizer):
        model.train()
        size = len(dataloader.dataset)
        for batch, (X, Y) in enumerate(dataloader):
            
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            pred = model(X)
            loss = cost(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss, current = loss.item(), batch * batch_size
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

    def test(dataloader, model, epoch):
        global bestAcc, bestEpoch

        model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for batch, (X, Y) in enumerate(dataloader):
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                pred = model(X)

                test_loss += cost(pred, Y).item()
                correct += (pred.argmax(1)==Y).type(torch.float).sum().item()
        
        size = len(dataloader.dataset)
        test_loss /= size
        correct /= size

        if correct > bestAcc:
            bestAcc = correct
            bestEpoch = epoch

        print(f'\nTest Error:\nacc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}, best acc: {(100*bestAcc):>0.1f}% at epoch {bestEpoch+1}\n')

    #split data to test and train
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print("Training size:", len(train_dataset))
    print("Testing size:",len(test_dataset))

    # labels in training set
    train_classes = [label for _, label in train_dataset]
    print(Counter(train_classes))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device)) 

    model = CNNet().to(device, non_blocking=True)
    # used to create optimal parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # state_dict = torch.load("model.pth")
    # model.load_state_dict(state_dict)

    start_time = time.time()

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, model, cost, optimizer)
        test(test_dataloader, model, t)
        end_time = time.time()
        print(f'Took: {end_time - start_time}s\n')
        start_time = end_time
        print('-------------------------------')
        if bestAcc * 100 >= stop_over:
            break

    print('Done!')

    summary(model, input_size=(batch_size, 1, target_num_samples))

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        correct = 0
        incorrect = 0
        for batch, (X, Y) in enumerate(test_dataloader):
            print("Batch: ", batch)
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            pred = model(X)
            
            for i in range(len(pred)):
                predicted = dataset.classes[pred[i].argmax(0)]
                actual = dataset.classes[Y[i]]
                print("Predicted: {}, Actual: {}\n".format(predicted, actual))
                if predicted == actual:
                    correct += 1
                else:
                    incorrect += 1
                # print("Predicted:\nvalue={}, class_name= {}\n".format(pred[i].argmax(0),dataset.classes[pred[i].argmax(0)]))
                # print("Actual:\nvalue={}, class_name= {}\n".format(Y[i],dataset.classes[Y[i]]))
        
        print("Final correct: ", correct, ", Incorrect: ", incorrect)

    torch.save(model.state_dict(), "trained.pth")
