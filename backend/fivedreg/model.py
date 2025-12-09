import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

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
    - max_it(int): The maximum number of iterations/epochs, default is 200
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

  def fit(self, X, y, batch_size=64, verbose=1, eval_every=10):
    """
    This function is to fit the model to the data
    Optimized for CPU training on datasets up to 10,000 samples within 1 minute

    Parameters:
    - X (array-like): The input data (i.e. X_train)
    - y (array-like): The target data (i.e. y_train)
    - batch_size (int): The batch size for mini-batch gradient descent, default is 64
    - verbose (int): Verbosity level, 0=no output, 1=progress every eval_every epochs
    - eval_every (int): Frequency of evaluation during training, default is every 10 epochs

    """
    if verbose:
      start_time = time.time()
    
    # Convert to PyTorch tensors (optimized for CPU)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Use mini-batches for efficiency
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    cost = nn.MSELoss()

    self.train()  # set model to training mode
    
    if verbose:
      print(f"Training on {len(X)} samples for {self.max_it} epochs...")
    
    for epoch in range(self.max_it):
      epoch_loss = 0.0
      batch_count = 0
      
      for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = self.model(X_batch)
        loss = cost(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
      
      # Log progress if verbose - will only show in backend
      if verbose and (epoch + 1) % eval_every == 0:
        avg_loss = epoch_loss / batch_count
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{self.max_it} - Loss: {avg_loss:.4f} - Time: {elapsed:.2f}s")
    
    if verbose:
      total_time = time.time() - start_time
      print(f"Training completed in {total_time:.2f} seconds")
      if total_time > 60:
        print(f" Warning: Training took {total_time:.2f}s (>60s target for <10k samples)")
      else:
        print(f"✓ Performance target met ({total_time:.2f}s < 60s)")
    
    return total_time
        

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
   y_pred = y_pred.numpy().flatten()
   return y_pred

