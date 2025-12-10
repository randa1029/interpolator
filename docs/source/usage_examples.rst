4. Usage Examples
=================

4.1 Example for Model Training and Prediction
----------------------------------------------
This is a simple demonstration with dummy data to illustrate how to train and predict with the model FiveDRegressor.

.. code-block:: python

    import numpy as np
    from backend.fivedreg.model import FiveDRegressor
    from torch import nn

    # Generate dummy data
    X_train = np.random.rand(100, 5)  # 100 samples, 5 features
    y_train = np.random.rand(100)      # 100 target values

    X_test = np.random.rand(10, 5)     # 10 samples for testing

    # Initialize the model
    model = FiveDRegressor(input_size = 5, output_size = 1, hidden_layers = [32], activation=nn.ReLU, lr = 1e-3, max_it = 200)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    print("Predictions:", predictions)


4.2 Example to Access Specific API Endpoints
---------------------------------------------
This example demonstrates how to access specific API endpoints in terminal.

.. code-block:: bash

    # Access the API endpoint for data upload
    curl -X 'POST' \
      'http://localhost:8000/upload' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'file=@coursework_dataset.pkl'
    
    # Access the API endpoint to preview dataset
    curl -X 'POST' \
      'http://localhost:8000/preview' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "filename": "coursework_dataset.pkl"
    }'

    # Access the API endpoint to train the model
    curl -X 'POST' \
      'http://localhost:8000/train' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "filename": "coursework_dataset.pkl",
      "hidden_layers": [
        64,
        32,
        16
      ],
      "lr": 0.001,
      "max_it": 200,
      "batch_size": 64,
      "test_size": 0.2,
      "val_size": 0.1
    }'

    # Access the API endpoint to make predictions
    curl -X 'POST' \
      'http://localhost:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "inputs": [
        [1, 2, 3, 4, 5]
      ]
    }'

    