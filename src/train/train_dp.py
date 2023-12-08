import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    '''
    prepare dataset for regression, default boston dataset
    '''

    # def __init__(self, X, y, scale_data = True):
    def __init__(self, X, y):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # # apply scaling if necessary
            # if scale_data:
            #     X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class MLP(nn.Module):
    '''
    Multiplayer Perceptron for regression
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        '''
            forward pass
        '''
        return self.layers(x)

class Trainer:
    # def __init__(self, model, train_data, val_data, lr=0.01, epochs=100):
    def __init__(self, model, train_data, lr=0.01, epochs=100):
        self.model = model
        self.train_data = train_data
        self.lr = lr
        self.epochs = epochs
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
    def train(self):
        train_loader = DataLoader(self.train_data, batch_size=32)
        # val_loader = DataLoader(self.val_data, batch_size=32)
        
        for epoch in tqdm(train_loader):
            running_loss = 0
            
            for X, y in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X) 
                loss = self.criterion(y_pred.squeeze(), y)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                
            # val_loss = 0
            # self.model.eval()
            # with torch.no_grad():
            #     for X, y in val_loader:
            #         y_pred = self.model(X)
            #         loss = self.criterion(y_pred.squeeze(), y)
            #         val_loss += loss.item()
            
            # print(f'Epoch: {epoch}, Training loss: {running_loss/len(train_loader):.3f}, Validation loss: {val_loss/len(val_loader):.3f}')

class Tester:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
        self.criterion = nn.MSELoss()
        
    def test(self):
        test_loader = DataLoader(self.test_data, batch_size=32) 
        test_loss = 0
        self.model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                y_pred = self.model(X) 
                loss = self.criterion(y_pred.squeeze(), y)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test loss: {test_loss:.3f}')


### load dataset
file = './src/data/winequality-red.csv'
df = pd.read_csv(file)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

## feature engineering
# Train - Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)
# Split train into train-val
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print(type(X_train))
print(type(y_train))
# model training & testing
train_ds = Dataset(X_train, y_train.to_numpy())
test_ds = Dataset(X_test, y_test.to_numpy())
model = MLP()
trainer = Trainer(model, train_ds)  
tester = Tester(model, test_ds)


trainer.train() 
tester.test()