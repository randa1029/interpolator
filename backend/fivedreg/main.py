from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import pickle
import shutil
import os
from .data import load_data, split_data
from .model import FiveDRegressor
import numpy as np

app = FastAPI()

# Global variable to store trained model
trained_model = None

#------------------------------------------------
# Root Endpoint
#------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Welcome to the FiveDRegressor API"}

#------------------------------------------------
# Health Check Endpoint
#------------------------------------------------
@app.get("/health")
async def get_health(): #keeping all endpoints asyncrhonous for consistency
    """
    Service health check endpoint.
    """
    return {"status": "healthy"}


#------------------------------------------------
# File Upload Endpoint
#------------------------------------------------

# Create an uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    #username: str = Form(None),
    #description: str = Form(None) - not including these fields as they have caused 422 validation erorr when tesing
):
    """
    Endpoint to upload a .pkl file
    """
    # --- 1. Validate File Extension (The required change) ---
    if not file.filename.lower().endswith(".pkl"):
        # Explicitly raise a 400 Bad Request error if the format is incorrect
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only files with the .pkl extension are accepted."
        )
    
    # --- 2. Secure File Saving (The required change) ---
    # 1. Extract only the filename, removing any user-provided path components (SECURITY FIX)
    safe_filename = os.path.basename(file.filename)
    
    # 2. Securely join the directory and the safe filename
    file_location = os.path.join(UPLOAD_DIR, safe_filename)

    # Save uploaded file locally
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully!",
        "filename": safe_filename,
        #"uploaded_by": username,
        #"description": description
    }
#------------------------------------------------
# Preview Pickle File Endpoint
#------------------------------------------------
class PreviewRequest(BaseModel):
    filename: str

@app.post("/preview")
async def preview_dataset(request: PreviewRequest):
    """
    Preview the first few rows of an uploaded .pkl dataset.
    """
    filename = request.filename
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Assume your pickle contains (X, y) or a dict
        if isinstance(data, tuple) and len(data) >= 2:
            X, y = data[0], data[1]
        elif isinstance(data, dict):
            # More general
            X = data.get("X")
            y = data.get("y")
        else:
            X = data
            y = None

        X = np.array(X)
        head = X[:5].tolist()

        return {
            "shape": list(X.shape),
            "head": head,
            "y_preview": y[:5].tolist() if y is not None else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")



#------------------------------------------------
# Model Training Endpoint
#------------------------------------------------
class TrainingConfig(BaseModel):
    filename: str
    hidden_layers: list[int] = [64, 32, 16]
    lr: float = 1e-3
    max_it: int = 200
    batch_size: int = 64
    test_size: float = 0.2
    val_size: float = 0.1

@app.post("/train")
async def train_model(config: TrainingConfig):
    """
    Train the model on an uploaded dataset.
    
    Parameters:
    - filename: Name of the uploaded .pkl file
    - hidden_layers: List of integers representing neurons per hidden layer
    - lr: Learning rate
    - max_it: Maximum iterations/epochs
    - batch_size: Batch size for training
    - test_size: Proportion for test split
    - val_size: Proportion for validation split
    """
    global trained_model
    
    #validating data split (need to sum up to 1)
    if (config.test_size + config.val_size) >= 1.0:
        raise HTTPException(
            status_code=400,
            detail="The sum of 'test_size' and 'val_size' must be less than 1.0."
        )
    try:
        # Sanitize the filename from the Pydantic model to prevent Path Traversal 
        safe_filename = os.path.basename(config.filename)
        # Construct file path
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File '{config.filename}' not found. Please upload it first."
            )
        
        # Load and preprocess data
        X, y = load_data(file_path)
        X_train, X_test, X_val, y_train, y_test, y_val = split_data(
            X, y, 
            test_size=config.test_size, 
            val_size=config.val_size
        )
        
        # Initialize model
        input_size = X_train.shape[1]
        output_size = 1
        trained_model = FiveDRegressor(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=config.hidden_layers,
            lr=config.lr,
            max_it=config.max_it
        )
        
        # Train model
        trained_model.fit(X_train, y_train, batch_size=config.batch_size)
        
        return {
            "message": "Model trained successfully!",
            "config": {
                "input_size": input_size,
                "hidden_layers": config.hidden_layers,
                "learning_rate": config.lr,
                "max_iterations": config.max_it,
                "batch_size": config.batch_size
            },
            "data_split": {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "val_samples": len(X_val)
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


#------------------------------------------------
# Prediction Endpoint
#------------------------------------------------
class PredictionInput(BaseModel):
    inputs: list[list[float]]  # List of input vectors, each with 5 features

@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Make predictions on new input vectors.
    
    Parameters:
    - inputs: List of input vectors (each should have the same number of features as training data)
    
    Returns:
    - predictions: List of predicted values
    """
    global trained_model
    
    # Check if model is trained
    if trained_model is None:
        raise HTTPException(
            status_code=400,
            detail="No trained model available. Please train a model first using the /train endpoint."
        )
    
    try:
        # Convert input to numpy array
        import numpy as np
        X_new = np.array(data.inputs, dtype=float)
        
        # Validate input dimensions
        if X_new.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D input array, got {X_new.ndim}D array."
            )
        
        # Make predictions
        predictions = trained_model.predict(X_new)
        
        return {
            "predictions": predictions.tolist(),
            "num_samples": len(predictions)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


#------------------------------------------------
# To call backend from frontend
#------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
