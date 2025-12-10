1. API Reference
================

1.1 Data Handling
-----------------

.. autofunction:: fivedreg.data.load_data

.. autofunction:: fivedreg.data.split_data

1.2 Main API Endpoints
----------------------

.. automodule:: fivedreg.main
   :members: root, get_health, upload, preview_dataset, train_model, predict
   :undoc-members:

1.3 Model
---------

.. autoclass:: fivedreg.model.FiveDRegressor
   :members: __init__, fit, predict
   :undoc-members:
   :show-inheritance: