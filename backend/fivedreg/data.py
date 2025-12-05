import pickle
import numpy as np
from sklearn.model_selection import train_test_split

with open('backend/fivedreg/data.pkl', 'rb') as file:
    data = pickle.load(file)


