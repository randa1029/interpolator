3. Installation
===============

3.1 Prerequisites
-----------------
Before using the Interpolator package, ensure you have the following prerequisites installed:
- Docker
- Python 3.8 or higher (for local development and testing)

3.2 Installation Steps
----------------------
To install the Interpolator package, follow these steps:

1. Clone the Repository

   .. code-block:: bash

      git clone https://github.com/yourusername/interpolator.git
      cd interpolator

2. Create and Activate Virtual Environment

   .. code-block:: bash

      python3.12 -m venv <name_of_venv>
      source <name_of_venv>/bin/activate

3. Install Dependencies (ones specified in the 'Dependencies' section in pyproject.toml). This is to ensure all required packages to run the python scripts are installed. Do so by running:

   .. code-block:: bash

      pip install -e .

4. To run backend server locally (for development and testing)
    .. code-block:: bash
        uvicorn backend.fivedreg.main:app --reload
    
Then backend API is accessible at: http://localhost:8000 . 
5. To run tests
    .. code-block:: bash
        pytest -s backend/tests/*
    
More about tests in Test Suites section.

6. To test frontend locally (for development and testing)
    .. code-block:: bash
        cd frontend
        npm install
        npm run dev
    
Then frontend Next.js application is accessible at: http://localhost:3000 .

3.3 Docker Deployment
---------------------
If instead want to deploy package through Docker, refer back to User Guides section.


