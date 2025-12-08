from backend.fivedreg.model import FiveDRegressor
from backend.fivedreg.data import load_data
import pytest
import tempfile
import pickle
imort numpy as np

@pytest.fixture #create one fixture that can be reused for all tests
def sample_pkldata():
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

    #create a temporary file to store the pickle data
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    with open(temp.name, 'wb') as f:
        pickle.dump(data, f)
    
    return temp.name

def test_structure() -> None:
    """ Testing model structure and output dimensions"""

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    #y = np.random.rand(100)      # 100 target values

    # Initialize model
    model = FiveDRegressor(input_size=5, output_size=1, hidden_layers=[10, 20, 30], lr=0.01, max_it=50)
    
    #Check if resulting number of hidden layers is the same as specified
    expected_layers = (len([10,20,30])*2 +1)
    assert len(model.model) == expected_layers, "Model layers do not match the specified hidden layers"
    #layers = 3 hidden layers + 3 activation layers + 1 output layer = 7 layers

    # Model ouput
    y_pred = model.predict(X)
    assert y_pred.shape == (100,), "Predicted output should have the same number of samples as input"


def test_training(sample_pkldata) -> None:
    """ Testing if model training properly"""
    X,y = load_data(sample_pkldata)
    model = FiveDRegressor(input_size=5, output_size=1, hidden_layers=[10, 20], lr=0.01, max_it=20)
    total_time = model.fit(X, y, batch_size=16, verbose=1, eval_every=5)

    assert total_time > 0, "Training time should be positive"


def test_prediction(sample_pkldata) -> None:
    """ Testing model prediction output"""
    X,y = load_data(sample_pkldata)
    model = FiveDRegressor(input_size=5, output_size=1, hidden_layers=[10, 20], lr=0.01, max_it=20)
    model.fit(X, y, batch_size=16, verbose=0)

    y_pred = model.predict(X)

    assert y_pred.shape == (len(X),), "Predicted output should have the same number of samples as input"
    assert isinstance(y_pred, np.ndarray), "Predicted output should be a numpy array"
