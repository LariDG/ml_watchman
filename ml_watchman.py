"""

Author: L. Dorman-Gajic
"""


import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd

from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error


infile = "file.csv"
file = open(str(infile))
data = pd.read_csv(file)

data = data[0].str.split(',',expand=True)



seed = 150
np.random.seed(seed)






    



    

    

    




    

        

