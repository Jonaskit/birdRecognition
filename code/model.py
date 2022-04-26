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

if __name__ == '__main__':
    epochs = 1000
    batch_size = 15
    num_workers = 2
    learning_rate = 0.0001
    train_ratio = 0.8
    data_path = './birdclef-2022/spectrograms' #looking in subfolder train

    bestAcc = 0
    bestEpoch = 0

    dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([transforms.Resize((201,81)), transforms.ToTensor()])
    )
    print(dataset)
    print("\n {} Class category and index of the images: {}\n".format(len(dataset.classes), dataset.class_to_idx))
    # cost function used to determine best parameters
    cost = torch.nn.CrossEntropyLoss()

    class CNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
            # self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
            # self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
            self.conv4 = nn.Conv2d(32, 64, kernel_size=5)
            self.conv4_drop = nn.Dropout2d()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(51136, 100)
            # self.fc2 = nn.Linear(2000, 50)
            # self.fc3 = nn.Linear(50, 50)
            self.fc4 = nn.Linear(100, len(dataset.classes))

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            # x = F.relu(F.max_pool2d(self.conv2(x), 2))
            # x = F.relu(F.max_pool2d(self.conv3(x), 2))
            x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))
            #x = x.view(x.size(0), -1)
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            # x = F.relu(self.fc2(x))
            # x = F.dropout(x, training=self.training)
            # x = F.relu(self.fc3(x))
            # x = F.dropout(x, training=self.training)
            x = F.relu(self.fc4(x))
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

    torch.save(model.state_dict(), "trained.pth")
