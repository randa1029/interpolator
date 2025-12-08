import numpy as np
import pytest
import pickle
import os
from backend.fivedreg.data import load_data, split_data


@pytest.fixture #create one fixture that can be reused for all tests
def sample_pkldata(tmp_path):
    """ Fixture to provide sample pickle data for testing"""
    X = np.array([[1.0, np.nan, 3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0, 9.0, 10.0],
                  [11.0, 12.0, 13.0, 14.0, 15.0],
                  [1.0, 2.0, 3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0, 9.0, 10.0],
                  [11.0, 12.0, 13.0, 14.0, 15.0],
                  [1.0, 2.0, 3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0, 9.0, 10.0],
                  [11.0, 12.0, 13.0, 14.0, 15.0],
                  [20.0, 21.0, 22.0, 23.0, 24.0]])
    
    # Matching 10 samples for y
    y = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    data = {'X': X, 'y': y}

    #use temp_path to creat a file path
    tmp_file_path = tmp_path / 'sample_data.pkl'
    with open(tmp_file_path, 'wb') as f:
        pickle.dump(data, f)
    
    return str(tmp_file_path) #return the file path as string

def test_data_handling(sample_pkldata) -> None: 
    """ Testing input data dimensions, and testing missing values handling"""

    #testing dimensions of loaded data
    X, y = load_data(sample_pkldata) 
    assert X.shape[1] == 5, "Input data should have 5 features"
    assert X.ndim == 2, "Input data should be two-dimensional"
    assert y.ndim == 1, "Output data should be one-dimensional"
    assert len(X) == len(y), "Input and output data should have the same number of samples"
    
  
    #testing missing values handling
    #1. check if output has NaN values
    assert not np.isnan(X).any(), "Input data should not contain NaN values after preprocessing"
    assert not np.isnan(y).any(), "Output data should not contain NaN values after preprocessing"
    
    #2. check if all NaN values are replaces by column means
    assert X[0,1] == np.nanmean(X[:,1]), "Missing value in X should be replaced by column mean"
    assert y[1] == np.nanmean(y), "Missing value in y should be replaced by mean of y"

def test_data_splitting(sample_pkldata) -> None:
    """ Testing data splitting and standardisation"""

    X, y = load_data(sample_pkldata) 
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y, test_size=0.2, val_size=0.1, random_state=42)

    # Check split sizes
    total_samples = len(X)
    assert len(X_train) == len(X) * 0.7, "Training set should be 70% of total data"
    assert len(X_test) == len(X) * 0.2, "Testing set should be 20% of total data"
    assert len(X_val) == len(X) * 0.1, "Validation set should be 10% of total data"
    assert len(X_train) + len(X_test) + len(X_val) == total_samples, "Total samples after split should match original data"

    # Check standardisation (mean ~0, std ~1)
    np.testing.assert_allclose(np.mean(X_train), 0, atol=1e-1, err_msg="Training data should be standardised to mean ~0")
    np.testing.assert_allclose(np.std(X_train), 1, atol=1e-1, err_msg="Training data should be standardised to std ~1")
    np.testing.assert_allclose(np.mean(X_test), 0, atol=1e-1, err_msg="Testing data should be standardised to mean ~0")
    np.testing.assert_allclose(np.std(X_test), 1, atol=1e-1, err_msg="Testing data should be standardised to std ~1")
    np.testing.assert_allclose(np.mean(X_val), 0, atol=1e-1, err_msg="Validation data should be standardised to mean ~0")
    np.testing.assert_allclose(np.std(X_val), 1, atol=1e-1, err_msg="Validation data should be standardised to std ~1") 