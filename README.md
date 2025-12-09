# interpolator
This is a package to build a system to learn and serve a light neural network model for 5D numerical dataset interpolation.

## Getting Started
---
**Prerequisites**
- Docker

1) To start, run shell script:<br>
`./scripts/docker-start.sh`

2) Access frontend via: http://localhost:3000 <br>
   Access backend via: http://localhost:8000

3) To stop, run:<br>
`./script/docker-end.sh`

- To see logs, run:<br>
`./scripts/docker-logs.sh/`

## Usage
---
This application allows users to:<br>
1) Upload .pkl files with 5D data, with the option to preview dataset (this is the dataset used for training)
2) Configure hyperparameters and train dataset. Configurable hyperparameters include: number of layers, unit per layer, learning rate, and maximum iterations. The optimiser used is set to Adam, default batch size is 64, and loss function is MSE.
3) Input values and predict result (i.e. interpolation)


## Environment Variables
---
**- In Backend Dockerfile:**
>`ENV PYTHONPATH=/app ` <br>
 This is specifying the system to look for additional python Packages in the additional directory `/app`

>`ENV PYTHONDONTWRITEBYTECODE=1` <br>
 This keeps Python from writing .pyc files. 

>`ENV PYTHONUNBUFFERED=1` <br>
This forces stdin, stdout, and stderr to be totally unbuffered. 

**- In Frontend Dockerfile:**
>`ENV NODE_ENV PRODUCTION` <br>
This tells Node.js to run in production mode.

>`ENV NEXT_TELEMETRY_DISABLED 1` <br>
This is to opt out telemetry collection by Next.js

>`ENV PORT 3000` <br>
This is to set environment variable `PORT` to tell the web server to listen on `3000`

>`ENV HOSTNAME "0.0.0.0"` <br>
This is to set environment variable `HOSTNAME` to be labelled as `0.0.0.0`

- **In *docker-compose.yml***
>`PYTHONPATH=/app` <br>
(same as above)

>`PYTHONUNBUFFERED=1`<br>
(same as above)

>`NODE_ENV=production`<br>
(same as above)

>`NEXT_TELEMETRY_DISABLED=1`<br>
(same as above)

>`NEXT_PUBLIC_API_URL=http://localhost:8000`<br>
This is to hold the address (`http://localhost:8000`) that the frontend application uses to make API requests


## Tests
---
The tests in `backend/tests` are unit tests to test validity of each function.

To run tests:<br>
- With Docker (run in quiet mode): 
```bash
docker compose exec backend pytest -q
```

- Run locally: <br>
```bash
pytest -s backend/tests/*
```
