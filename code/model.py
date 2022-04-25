from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torchinfo import summary
import pandas as pd
import os
from collections import Counter
import time

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, len(dataset.classes))


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x,dim=1) 

def train(dataloader, model, loss, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


# Create the validation/test function

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1)==Y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(f'\nTest Error:\nacc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')

if __name__ == '__main__':
    epochs = 150
    data_path = './birdclef-2022/spectrograms' #looking in subfolder train

    dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([transforms.Resize((201,81)),
                                    transforms.ToTensor()
                                    ])
    )
    print(dataset)

    class_map=dataset.class_to_idx

    print("\n {} Class category and index of the images: {}\n".format(len(dataset.classes), class_map))

    #split data to test and train
    #use 80% to train
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print("Training size:", len(train_dataset))
    print("Testing size:",len(test_dataset))

    # labels in training set
    train_classes = [label for _, label in train_dataset]
    print(Counter(train_classes))

    batch_size = 40
    num_workers = 2
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

    td = train_dataloader.dataset[0][0][0][0]
    print(td)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device)) 

    model = CNNet().to(device, non_blocking=True)

    # cost function used to determine best parameters
    cost = torch.nn.CrossEntropyLoss()

    # used to create optimal parameters
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create the training function

    start_time = time.time()

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, model, cost, optimizer)
        test(test_dataloader, model)
        end_time = time.time()
        print(f'Took: {end_time - start_time}s\n')
        start_time = end_time
        print('-------------------------------')

    print('Done!')

    summary(model, input_size=(15, 3, 201, 81))

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
