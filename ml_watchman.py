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


mc_infile = sys.argv[1]
reco_infile = sys.argv[2]


mc = open(str(mc_infile))
mc_read = pd.read_csv(mc)
mc_data = mc_read[["gtid", "mcx", "mcy", "mcz", "mc_energy", "closestPMT",  "n100",  "true_wall_r",  "true_wall_z"]]

reco = open(str(reco_infile))
reco_read = pd.read_csv(reco)
reco_data = reco_read[["gtid", "x", "y", "z", "pe", "closestPMT_prev", "n100_prev", "reco_wall_r", "reco_wall_z"]]




seed = 150
np.random.seed(seed)


E_low = 0
E_high = 10
div = 100
bins = int((E_high-E_low)/div)

# assert(dfsel.isnull().any().any()==False) for checking nan values (from Lilia's code)

print(mc_data.)
print(reco_data)







    



    

    

    




    

        

