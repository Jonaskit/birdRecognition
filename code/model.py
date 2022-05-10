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
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve



if __name__ == '__main__':

    epochs = 200
    dropout = False
    batch_size = 15
    num_workers = 2
    learning_rate = 0.0001
    train_ratio = 0.8
    stop_over = 90 # percent
    data_path = './birdclef-2022/spectrograms' #looking in subfolder train
    eval_losses=[]
    eval_accu=[]
    train_accu=[]
    train_losses=[]
    bestAcc = 0
    bestEpoch = 0

    dataset = datasets.ImageFolder(
        root=data_path,
#        transform=transforms.Compose([transforms.Resize((201,81)), transforms.ToTensor()])
        transform=transforms.Compose([transforms.Resize((201,481)), transforms.ToTensor()])
    )
    print(dataset)
    print("\n {} Class category and index of the images: {}\n".format(len(dataset.classes), dataset.class_to_idx))
    # cost function used to determine best parameters
    cost = torch.nn.CrossEntropyLoss()

    class CNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=2),
                #torch.nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
                #torch.nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
                #torch.nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
                #torch.nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2),
                #torch.nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            self.flatten = nn.Flatten()
            self.linear = nn.Linear(32768, 2000)
            self.linear2 = nn.Linear(2000, len(dataset.classes))
            self.softmax = nn.Softmax(dim=1)
            self.dropout = nn.Dropout(0.3)

        def forward(self, input_data):
            x = self.conv1(input_data)
            x = self.conv2(x)
            if dropout:
                x = self.dropout(x)
            x = self.conv3(x)
            if dropout:
                x = self.dropout(x)
            x = self.conv4(x)
            if dropout:
                x = self.dropout(x)
            x = self.conv5(x)

            x = self.flatten(x)
            logits = self.linear(x)
            logits = self.linear2(logits)
            predictions = self.softmax(logits)
            return predictions

    def progress_bar(current, total, bar_length=20):
        fraction = current / total
        arrow = int(fraction * bar_length - 1) * '-' + '>'
        padding = int(bar_length - len(arrow)) * ' '
        ending = '\n' if current == total else '\r'
        print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', ending)

    def train(dataloader, model, cost, optimizer):
        model.train()
        running_loss=0
        correct=0
        total=0

        
        size = len(dataloader.dataset)
        for batch, (X, Y) in enumerate(dataloader):
            
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            pred = model(X)
            loss = cost(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item, current = loss.item(), (batch + 1) * batch_size
            running_loss += loss_item
            
            _, predicted = pred.max(1)
            total += Y.size(0)
            correct += predicted.eq(Y).sum().item()
            
            progress_bar(current if current < size else size, size)

        train_loss=running_loss/len(dataloader)
        accu=100.*correct/total
  
        train_accu.append(accu)
        train_losses.append(train_loss)
        print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

        
    def test(dataloader, model, epoch):
        global bestAcc, bestEpoch
 
        
        model.eval()
        test_loss = 0 
        correct = 0
        running_loss = 0 
        total = 0

        with torch.no_grad():
            for batch, (X, Y) in enumerate(dataloader):
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                pred = model(X)
                loss = cost(pred, Y) 
                running_loss += loss.item()
                
                _, predicted = pred.max(1)
                total += Y.size(0)
                correct += predicted.eq(Y).sum().item()
        
                
        test_loss=running_loss/len(dataloader)
        accu=100.*correct/total

        eval_losses.append(test_loss)
        eval_accu.append(accu)

        print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
        
        size = len(dataloader.dataset)
        test_loss /= size
        correct /= size

        if correct > bestAcc:
            bestAcc = correct
            bestEpoch = epoch
      #  y_pred = np.array(pred)
      #  print("Y_pred:", y_pred)
      #  y_test = np.array(Y)
      #  print("y_test:", y_test)




    #split data to test and train
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print("Training size:", len(train_dataset))
    print("Testing size:",len(test_dataset))

    # labels in training set
  #  train_classes = [label for _, label in train_dataset]
  #  print(Counter(train_classes))

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
        print(f'Took: {end_time - start_time}s')
        start_time = end_time
        print('-------------------------------')
        if bestAcc * 100 >= stop_over:
            break

    print('Done!')

    summary(model, input_size=(15, 3, 201, 481))

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        correct = 0
        incorrect = 0
        
        y_pred = np.zeros(test_size)
        y_test = np.zeros(test_size)
        count = 0
        for batch, (X, Y) in enumerate(test_dataloader):
            print("Batch: ", batch)
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            pred = model(X)
            
            for i in range(len(pred)):
                predicted = dataset.classes[pred[i].argmax(0)]
                pred_index = dataset.class_to_idx.get(dataset.classes[pred[i].argmax(0)])
                actual = dataset.classes[Y[i]]
                actual_index = dataset.class_to_idx.get(dataset.classes[Y[i]])

                y_pred[count] = pred_index
                y_test[count] = actual_index
                count = count + 1
               # print("Predicted: {}, Actual: {}\n".format(predicted, actual))
                if predicted == actual:
                    correct += 1
                else:
                    incorrect += 1
                # print("Predicted:\nvalue={}, class_name= {}\n".format(pred[i].argmax(0),dataset.classes[pred[i].argmax(0)]))
                # print("Actual:\nvalue={}, class_name= {}\n".format(Y[i],dataset.classes[Y[i]]))
      #  y_pred = np.array(dataset.classes[pred.argmax(0)])
      #  y_test = np.array(dataset.classes[Y])
      
        #labels = ["1", "2", "3", "4", "5"]

      
        cm = confusion_matrix(y_test, y_pred)
        
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        print("Final correct: ", correct, ", Incorrect: ", incorrect)
        
        #accuracy plots
        plt.plot(train_accu,'-o')
        plt.plot(eval_accu,'-o')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Accuracy')

        plt.show()
        
        
        #loss plot
        plt.plot(train_losses,'-o')
        plt.plot(eval_losses,'-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Losses')
        
        plt.show()
        
        
    torch.save(model.state_dict(), "trained.pth")

