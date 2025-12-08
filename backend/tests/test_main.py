import numpy as np
import pytest
import pickle
import tempfile
import os
from unittest.mock import patch
from fivedreg.main import app
from fastapi.testclient import TestClient


client = TestClient(app)

@pytest.fixture
def temp_upload_dir():
    """ Fixture to create a temporary upload directory for tests """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup: remove all files in temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_pkldata():
    """ Fixture to provide sample pickle data for testing """
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

    # Create a temporary file to store the pickle data
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    with open(temp.name, 'wb') as f:
        pickle.dump(data, f)
    
    yield temp.name
    
    # Cleanup the fixture file
    if os.path.exists(temp.name):
        os.unlink(temp.name)

##################Tests####################

#Testing root
def test_root():
    """ Testing root endpoint """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "FiveDRegressor API is running"}

#Testing health
def test_health():
    """ Testing health endpoint """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Testing file upload endpoint - success and failure cases
def test_upload_success(sample_pkldata, temp_upload_dir):
    """ Testing successful file upload endpoint """
    
    with patch('fivedreg.main.UPLOAD_DIR', temp_upload_dir):
        with open(sample_pkldata, 'rb') as f:
            response = client.post("/upload", files={"file": f})
        
        assert response.status_code == 200
        assert "message" in response.json()
        assert "filename" in response.json()


def test_upload_failure():
    """ Testing failed file upload endpoint """
    
    # Create a non-pkl file
    temp_txt = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    temp_txt.write(b"not a pickle file")
    temp_txt.close()
    
    with open(temp_txt.name, 'rb') as f:
        response = client.post("/upload", files={"file": f})
    
    os.unlink(temp_txt.name)
    
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


# Testing preview endpoint 
def test_preview_endpoint(sample_pkldata, temp_upload_dir):
    """ Testing preview pickle file endpoint """
    
    with patch('fivedreg.main.UPLOAD_DIR', temp_upload_dir):
        # First upload the fixture file
        with open(sample_pkldata, 'rb') as f:
            upload_response = client.post("/upload", files={"file": f})
        
        assert upload_response.status_code == 200
        filename = upload_response.json()["filename"]
        
        # Now preview the uploaded file
        preview_response = client.post("/preview", json={"filename": filename})
        
        assert preview_response.status_code == 200
        data = preview_response.json()
        
        # Check response structure
        assert "shape" in data
        assert "head" in data
        assert "y_preview" in data
        
        # Check data values from our fixture (10 samples, 5 features)
        assert data["shape"] == [10, 5]
        assert len(data["head"]) == 5  # First 5 rows
        assert data["y_preview"] is not None



# Testing model training endpoint
def test_training_endpoint(sample_pkldata, temp_upload_dir):
    """ Testing model training endpoint """
    
    with patch('fivedreg.main.UPLOAD_DIR', temp_upload_dir):
        # First upload the file
        with open(sample_pkldata, 'rb') as f:
            upload_response = client.post("/upload", files={"file": f})
        
        assert upload_response.status_code == 200
        filename = upload_response.json()["filename"]
        
        # Now train the model
        train_payload = {
            "filename": filename,
            "hidden_layers": [10, 20],
            "lr": 0.01,
            "max_it": 10,
            "batch_size": 4,
            "test_size": 0.2,
            "val_size": 0.1
        }
        
        train_response = client.post("/train", json=train_payload)
        
        assert train_response.status_code == 200
        data = train_response.json()
        
        # Check response structure
        assert "message" in data
        assert "config" in data
        assert "data_split" in data
        assert data["config"]["hidden_layers"] == [10, 20]
        assert data["data_split"]["train_samples"] > 0


def test_training_file_not_found():
    """ Testing training endpoint with non-existent file """
    
    train_payload = {
        "filename": "nonexistent.pkl",
        "hidden_layers": [10],
        "lr": 0.01,
        "max_it": 10,
        "batch_size": 4,
        "test_size": 0.2,
        "val_size": 0.1
    }
    
    response = client.post("/train", json=train_payload)
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# Testing prediction endpoint
def test_prediction_endpoint(sample_pkldata, temp_upload_dir):
    """ Testing model prediction endpoint - full workflow """
    
    with patch('fivedreg.main.UPLOAD_DIR', temp_upload_dir):
        # Step 1: Upload the file
        with open(sample_pkldata, 'rb') as f:
            upload_response = client.post("/upload", files={"file": f})
        
        assert upload_response.status_code == 200
        filename = upload_response.json()["filename"]
        
        # Step 2: Train the model
        train_payload = {
            "filename": filename,
            "hidden_layers": [10],
            "lr": 0.01,
            "max_it": 5,
            "batch_size": 4,
            "test_size": 0.2,
            "val_size": 0.1
        }
        
        train_response = client.post("/train", json=train_payload)
        assert train_response.status_code == 200
        
        # Step 3: Make predictions
        predict_payload = {
            "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0], 
                       [6.0, 7.0, 8.0, 9.0, 10.0]]
        }
        
        predict_response = client.post("/predict", json=predict_payload)
        
        assert predict_response.status_code == 200
        data = predict_response.json()
        
        # Check response structure
        assert "predictions" in data
        assert "num_samples" in data
        assert len(data["predictions"]) == 2
        assert data["num_samples"] == 2
        assert all(isinstance(p, (int, float)) for p in data["predictions"])


def test_prediction_no_model():
    """ Testing prediction endpoint without training first """
    
    # Reset trained_model to None (simulate fresh start)
    from fivedreg import main
    main.trained_model = None
    
    predict_payload = {
        "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0]]
    }
    
    response = client.post("/predict", json=predict_payload)
    
    assert response.status_code == 400
    assert "No trained model available" in response.json()["detail"]


def test_prediction_invalid_dimensions():
    """ Testing prediction endpoint with wrong input dimensions """
    
    # This test assumes a model was trained in previous test
    # If running in isolation, you may need to train first
    
    predict_payload = {
        "inputs": [[1.0, 2.0]]  # Only 2 features instead of 5
    }
    
    response = client.post("/predict", json=predict_payload)
    
    # This should fail during prediction due to dimension mismatch
    assert response.status_code in [400, 500]  # Could be either depending on where it fails

