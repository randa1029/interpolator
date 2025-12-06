import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class FiveDRegressor(nn.Module):
  def __init__(self, input_size: int, output_size: int, hidden_layers:list, activation=nn.ReLU, lr = 1e-3, max_it = 200):
    """
    This function is to implement a configurable lightweight neural network

    Parameters:
    - input_size (int) : The number of input features (i.e. 5 for the 5D dataset), 
    - output_size (int): The number of output features, 
    - hidden_layers(list): A list of integers, number of integers correspons to number of layers, and each number shows the number of neurons per layer, 
    - activation: The activation function to use, default is ReLU
    - lr(float): The learning rate, default is 1e-3
    - max_it(int): The maximum number of iterations

    Returns:


    """
    super().__init__()
    self.lr = lr
    self.max_it = max_it

    layers = []
    in_dim = input_size

    #hidden layers
    for i in hidden_layers:
      layers.append(nn.Linear(in_dim, i))
      layers.append(activation())
      in_dim = i

    #output layer
    layers.append(nn.Linear(in_dim, output_size))
    
    #build model
    self.model = nn.Sequential(*layers)

  def fit(self,X,y, batch_size = 64):
    """
    This function is to fit the model to the data

    Parameters:
    - X (array-like): The input data (i.e. X_train)
    - y (array-like): The target data (i.e. y_train)

    """
    #convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1,1)

    #use mini-batches
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)

    optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    loss = nn.MSELoss()

    self.train() #set model to training model
    for epoch in range(self.max_it):
      for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = self.model(X)
        loss = loss(y_pred, y)
        loss.backward()
        optimizer.step()

  def predict(self,X):
   """
   This function is to predict output of the model

   Parameters:
   - X (array-like): The input data (i.e. X_test)

   Returns:
   - y_pred (array-like): The predicted output
   """

   self.eval() #set model to evaluation mode

   X = torch.tensor(X, dtype=torch.float32)
   with torch.no_grad():
     y_pred = self.model(X)
  
   self.train() #set model back to training mode
   return y_pred.numpy().flatten()

