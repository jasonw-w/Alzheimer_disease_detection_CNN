import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random
import math
train_PATH = r'Alzheimer MRI Disease Classification Dataset\Data\train-00000-of-00001-c08a401c53fe5312.parquet'
train = pd.read_parquet(train_PATH)
test_PATH = r'Alzheimer MRI Disease Classification Dataset/Data/test-00000-of-00001-44110b9df98c5585.parquet'
test = pd.read_parquet(test_PATH)
disease_label_from_category = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non Demented",
    3: "Very Mild Demented",
}

def dict_to_image(image_dict):
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        byte_string = image_dict['bytes']
        nparr = np.frombuffer(byte_string, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    else:
        raise TypeError(f"Expected dictionary with 'bytes' key, got {type(image_dict)}")


train['image_arr'] = train['image'].apply(dict_to_image)
test['image_arr'] = test['image'].apply(dict_to_image)

class AlzheimerDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img = torch.tensor(self.df.iloc[idx]['image_arr'], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.df.iloc[idx]['category'], dtype=torch.long)
        if self.transform:
            img = self.transform(img)
        return img, label

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__() # inherit methods from nn.Module
        #input shape = (1, 128, 128)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)#128-3+1 = 126
        self.maxpool1 = nn.MaxPool2d(2, 2)#126/2 = 63
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1)#63-4+1 = 60
        #self.maxpool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*60*60, 256)
        self.fc2 = nn.Linear(256, 4)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        #x = self.maxpool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

class Trainer:
    def __init__(self, model, lr, optim, loss_fn, train_loader, test_loader):
        self.model = model
        self.lr = lr
        self.optim = optim
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, epochs):
        plt.ion()
        plot_total = []
        plot = []
        for epoch in range(epochs):
            self.model.train()
            for batch, (img, label) in enumerate(self.train_loader):
                img, label = img.to(self.device), label.to(self.device)
                self.optim.zero_grad()
                output = self.model(img)
                loss = self.loss_fn(output, label)
                loss.backward()
                self.optim.step()
                pred = output.argmax(dim=1, keepdim=True)

                correct = (pred == label).all()
                if correct:
                    plot_total.append(1)
                else:
                    plot_total.append(0)
                model_loss = self.loss_fn(output, label)
                model_loss.backward()
                self.optimizer.step()
                plot.append(plot_total.sum()/len(plot_total))
            print(f"Epoch {epoch} Loss: {model_loss.item()}")
            plt.plot(plot)
            plt.show()
            plt.pause(0.1)
    def test(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            for batch, (img, label) in enumerate(test_loader):
                img, label = img.to(self.device), label.to(self.device)
                output = self.model(img)
                pred = output.argmax(dim=1, keepdim=True)
                correct = (pred == label).all()
                print(f"Accuracy: {correct.sum()/len(correct)}")

batch_size = 128
lr = 0.001
model = CNN_Model()
optim = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(AlzheimerDataset(train), batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(AlzheimerDataset(test), batch_size=batch_size, shuffle=True)
epochs = 50

trainer = Trainer(model, lr, optim, loss_fn, train_loader, test_loader)
model = CNN_Model()
trainer.train(epochs)
trainer.test(test_loader)


