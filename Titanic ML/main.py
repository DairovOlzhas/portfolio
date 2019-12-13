import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import Dataset
from model import NeuralNetwork

batch_size = 8
train_dataset = Dataset()

validation_split = 0.1
dataset_len = len(train_dataset)
indices = list(range(dataset_len))

val_len = int(np.floor(validation_split * dataset_len))
validation_idx = np.random.choice(indices, size=val_len, replace = False)
train_idx = list(set(indices) - set(validation_idx))

train_sampler = SubsetRandomSampler(train_idx) 
validation_sampler = SubsetRandomSampler(validation_idx) 

train_loader = DataLoader(train_dataset,
                         # shuffle=True,
                         batch_size=batch_size,
                         sampler=train_sampler)

val_loader = DataLoader(train_dataset,
                         # shuffle=True,
                         batch_size=batch_size,
                         sampler=validation_sampler)

data_loaders = {"train": train_loader, "val": val_loader}
data_lengths = {"train": len(train_idx), "val": val_len}


net = NeuralNetwork()
torch.save(net.state_dict(), './model.pth')

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(),
                      lr=0.001,
                      momentum=0.9)

nepoch = 100
for epoch in range(nepoch):
    # print('Epoch {}/{}'.format(epoch, nepoch - 1))
    # print('-' * 10)
    correct_val = 0.
    all_cnt_val = 0.
    correct_train = 0.
    all_cnt_train = 0.
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train(True)  # Set model to training mode
        else:
            net.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        for X, Y in data_loaders[phase]:
            optimizer.zero_grad()

            pred = net(X)
            # print(pred)
            if phase == 'train':
                all_cnt_train += len(pred)
                for i in range(len(pred)):
                    if (pred[i] >= 0.5 and Y[i] == 1) or (pred[i] < 0.5 and Y[i] == 0):
                        correct_train+=1

            else:
                all_cnt_val += len(pred)
                for i in range(len(pred)):
                    if (pred[i] >= 0.5 and Y[i] == 1) or (pred[i] < 0.5 and Y[i] == 0):
                        correct_val+=1


            loss = criterion(pred, Y)

            if phase == 'train':
                loss.backward()
                optimizer.step()


            running_loss += loss.item()
        epoch_loss = running_loss / data_lengths[phase]
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        if phase == 'val':
            print('Accuracy validation: {:.4f}%'.format(100-correct_val/all_cnt_val*100))
        if phase == 'train':
            print('Accuracy train: {:.4f}%'.format(100-correct_train/all_cnt_train*100))

torch.save(net.state_dict(), './model.pth')
