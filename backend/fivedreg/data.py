import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

#------------------------------------------------
# Load Data + Validate + Handling Missing Values
#------------------------------------------------

def load_data(filepath):
    """
    Load dataset from a pickle file, then validate input dimensions and handle missing values.

    Inputs:
    ----------
    filepath : str
        Path to the pickle file containing the dataset.

    Returns:
    ----------
    X : np.ndarray
        Feature matrix.
    
    y : np.ndarray
        Target vector.
    """
    #load file
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    
    X = np.array(data['X'], dtype=float)
    y = np.array(data['y'], dtype=float).reshape(-1)

    # Validate dimensions
    if X.shape[1] != 5:
        raise ValueError(f"Expected 5 features in X, got {X.shape[1]} features instead.")
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2-dimensional, got {X.ndim}-dimensional array instead.")
    if y.ndim != 1:
        raise ValueError(f"Expected y to be 1-dimensional, got {y.ndim}-dimensional array instead.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Number of samples in X ({X.shape[0]}) does not match number of samples in y ({y.shape[0]}).")

    # Handle missing values by replacing them with local means
    X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).to_numpy()
    y = pd.Series(y).fillna(pd.Series(y).mean()).to_numpy()
    return X, y


#------------------------------------------------
# Split Data and Standardise Features
#------------------------------------------------
def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the dataset into training and testing sets, then standardise the features.
    Here I chose to have 70% train, 20% test, and 10% validation splits.

    Inputs:
    ----------
    X : np.ndarray
        Feature matrix.
    
    y : np.ndarray
        Target vector.
    
    test_size : float
        Proportion of the dataset to include in the test split.
    
    random_state : int
        Random seed for reproducibility.

    Returns:
    ----------
    X_train : np.ndarray
        Standardised training feature matrix.
    X_test : np.ndarray
        Standardised testing feature matrix.
    y_train : np.ndarray
        Training target vector.
    y_test : np.ndarray
        Testing target vector.
    """
    ####################Split data
    #test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    #validation train split
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=relative_val_size, random_state=random_state)

    ####################Standardise features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) #only use transfrom here as we are applying the SAME standardisation as X_train
    X_val = scaler.transform(X_val)

    return X_train, X_test, X_val, y_train, y_test, y_val