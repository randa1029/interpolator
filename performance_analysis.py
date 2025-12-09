#This script is for question 8 to conduct perfomance analysis
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import profiler #this is a profiler API userful to identify time and memory costs of various PyTorch operations in the code
from torch.profiler import ProfilerActivity, record_function
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from backend.fivedreg.model import FiveDRegressor
from torchmetrics.regression import MeanSquaredError, R2Score

#import tqdm - do i need this and see progress bars??

###########################
#    Generate samples     #
###########################
def generate_samples(n_samples, n_features = 5, noise = 1, random_sate = 45):

    """Function to generate samples for performance analysis"""

    X, y = make_regression(n_samples = n_samples, n_features = n_features, noise = noise, random_state = random_sate)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size = 0.2, random_state = random_sate)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size = 0.25, random_state = random_sate) # 0.25 x 0.8 = 0.2
    return X, y, Xtr, Xval, Xte, ytr, yval, yte


###############################
#    Measure Training time    #
###############################
def measure_training_time(X, y):

    """
    Function to measure training time of FiveDRegressor on given data
    
    Parameters:
    -----------
    X: np.ndarray
        Input features (Train data)
    y: np.ndarray
        Target values (Train data)

    Returns:
    -----------
    training_time: float
        Time taken to train the model in seconds
    """

    model = FiveDRegressor(input_size = 5, output_size = 1, hidden_layers = [64,32,16], activation=nn.ReLU, lr = 1e-3, max_it = 200) #initialize model with default parameters,i.e. lr = 1e-3, max_it = 200

#----------1.Warm up runs
    for _ in range(10):
        _ = model.fit(X, y, verbose = 0)
    
    #Measure inference time
    start_time = time.time()
    output = model.fit(X, y,verbose = 0)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time after warm-up: {training_time:.4f} seconds")

    final_training_time = None
#----------2. Multiple iterations
#For more reliable measurement, run multiple iterations and take average
    num_iterations = 100
    training_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        output = model.fit(X, y,verbose = 0)
        end_time = time.time()
        training_times.append(end_time - start_time)
    average_training_time = np.mean(training_times)

#----------3. GPU Syncrhonization (if applicable) - which is NOT for my labtop  - CPU only
    
#----------4. Profiling with PyTorch Profiler
    with profiler.profile(record_shapes=True) as prof:
        output = model.fit(X, y, verbose = 0)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    return final_training_time



###############################
#   Profiling Memory Usage    #
###############################

def memory_usage(Xtr, ytr, Xte, yte):

    """
    Function to profile memory usage during both training and prediction phases to identify potential bottlenecks.
    
    Parameters:
    -----------
    Xtr: np.ndarray
        Input features (Train data) 

    ytr: np.ndarray
        Target values (Train data)

    Xte: np.ndarray
        Input features (Test data)

    yte: np.ndarray
        Target values (Test data)
    
    Returns:
    -----------
    memory_profile: profiler.profile
        Memory profile object containing memory usage details
    """

    model = FiveDRegressor(input_size = 5, output_size = 1, hidden_layers = [64,32,16], activation=nn.ReLU, lr = 1e-3, max_it = 5) #reduce number of epochs for profiling
#-----------------profiling during training
    #set up profile
    with torch.profile.profile(
        activities = [ProfilerActivity.CPU],
        record_samples = True,
        profile_memory = True) as prof_train:

        with record_function("model_training"): #record function allows memory using for training to be isolated, making it easier to analyse
            output = model.fit(Xtr, ytr, verbose = 0) #Forward pass
        
    print("Memory usage during training:")
    print(prof_train.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    
#-----------------profiling during prediction
    with torch.profile.profile(
        activities = [ProfilerActivity.CPU],
        record_samples = True,
        profile_memory = True) as prof_pred:

        with record_function("model_prediction"): #record function allows memory using for training to be isolated, making it easier to analyse
            output = model.predict(Xte, verbose = 0) #Forward pass
        
    print("Memory usage during prediction:")
    print(prof_pred.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    

    return prof_train, prof_pred



###############################
#       Accuracy Metrics      #
###############################
def accuracy_metrics(Xtr,ytr,Xte,yte):
    """
    Function to compute accuracy metrics - MSE and R^2 score on test data
    
    Parameters:
    -----------
    Xtr: np.ndarray
        Input features (Train data)

    ytr: np.ndarray
        Target values (Train data)

    Xte: np.ndarray
        Input features (Test data)

    yte: np.ndarray
        True target values on test data

    Returns:
    -----------
    mse: float
        Mean Squared Error on test data
    
    r2: float
        R^2 score on test data
    """

    model = FiveDRegressor(input_size = 5, output_size = 1, hidden_layers = [64,32,16], activation=nn.ReLU, lr = 1e-3, max_it = 200)
    
    model.fit(Xtr, ytr, verbose = 0)
    
    yte = torch.tensor(yte, dtype = torch.float32).view(-1,1)

    ypred = model.predict(Xte)
    ypred = torch.tensor(ypred, dtype = torch.float32).view(-1,1)

    mse = MeanSquaredError()
    mse_value = mse(ypred,yte)

    r2 = R2Score()
    r2_value = r2(ypred,yte)

    print(f"Mean Squared Error on test data: {mse_value.item():.4f}")
    print(f"R^2 Score on test data: {r2_value.item():.4f}")
    return mse_value.item(), r2_value.item()

    
