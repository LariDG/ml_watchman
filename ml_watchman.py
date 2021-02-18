"""

Author: L. Dorman-Gajic
"""
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd

from sklearn import svm, datasets
from sklearn import model_selection
from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


#opening files in command line, one for true parameters and one for the reconstructed
mc_infile = sys.argv[1]
reco_infile = sys.argv[2]

#random seed to improve reproducibility
seed = 150
np.random.seed(seed)

E_low = 0
E_high = 10
div = 100
bins = (E_high-E_low)/div

#reading files into data frame
mc = open(str(mc_infile))
print("true data: ",mc)
mc_read = pd.read_csv(mc)
df_mc = mc_read[['gtid', 'mcx', 'mcy', 'mcz', 'mc_energy', 'closestPMT',  'n100',  'true_wall_r',  'true_wall_z']]

reco = open(str(reco_infile))
reco_read = pd.read_csv(reco)
df_reco = reco_read[['gtid', 'x', 'y', 'z', 'pe', 'closestPMT_prev', 'n100_prev', 'reco_wall_r', 'reco_wall_z']]

#print(df_reco)

#normalising variables in files
df_reco_n = pd.DataFrame([df_reco['x']/10000, df_reco['y']/10000, df_reco['z']/10000,  df_reco['pe']/400, df_reco['closestPMT_prev']/7000, df_reco['n100_prev']/300, df_reco['reco_wall_r']/10000, df_reco['reco_wall_z']/10000]).T

print("check normalisation of reconstructed data:")
print(df_reco_n.head(), df_reco_n.shape)
print("----------------------------------------------------------------------------")


df_mc_n = pd.DataFrame([df_mc['mcx']/10000, df_mc['mcy']/10000, df_mc['mcz']/10000,  df_mc['mc_energy']/10, df_mc['closestPMT']/7000, df_mc['n100']/400, df_mc['true_wall_r']/10000, df_mc['true_wall_z']/10000]).T

print("check normalisation of true data: ")
print(df_mc_n.head(), df_mc_n.shape)
print("----------------------------------------------------------------------------")


mc_array = np.array(df_mc_n[['mc_energy']])

reco_array = np.array(df_reco_n[['x', 'y', 'z', 'pe', 'closestPMT_prev', 'n100_prev', 'reco_wall_r', 'reco_wall_z']])


rnd_indices = np.random.rand(len(reco_array)) < 0.50

mc_array_train = mc_array[rnd_indices]
reco_array_train = reco_array[rnd_indices]

mc_array_pred = mc_array[~rnd_indices]
reco_array_pred = reco_array[~rnd_indices]

print('number of training events and shape(true): ', len(mc_array_train), ' ', mc_array_train.shape,  ' number of predicting events and shape: ', len(mc_array_pred), ' ', mc_array_pred.shape)


print('number of training events and shape(reco): ', len(reco_array_train), ' ', reco_array_train.shape,  ' number of predicting events and shape: ', len(reco_array_pred), ' ', reco_array_pred.shape)


print('----------------------------------------------------------------------------')


n_estimators=1000
params = {'n_estimators':n_estimators, 'max_depth': 50,
    'learning_rate': 0.01, 'loss': 'lad'}

offset = int(reco_array_train.shape[0]*0.7)
reco_train, mc_train = reco_array_train[:offset], mc_array_train[:offset].reshape(-1)
reco_test, mc_test = reco_array_train[offset:], mc_array_train[offset:].reshape(-1)

print("train shape reco: ", reco_train.shape, "train shape mc: ", mc_train.shape)
print("test shape reco: ", reco_test.shape, "test shape mc: ", mc_test.shape)

gbr = ensemble.GradientBoostingRegressor(**params)
gbr.fit(reco_train, mc_train)

mse = mean_squared_error(mc_test, gbr.predict(reco_test))

print("mean squared error: %.4f" % mse)

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(gbr.staged_predict(reco_test)):
    test_score[i] = gbr.loss_(mc_test, y_pred)

pred_output = gbr.predict(reco_array_pred)
print(pred_output)

Y=[0 for j in range (0,len(mc_array_pred))]
for i in range(len(mc_array_pred)):
    Y[i] = 100 * (mc_array_pred[i] - pred_output[i]) / (1 * mc_array_pred[i])

df1 = pd.DataFrame(mc_array_pred,columns=['TrueEnergy'])
df2 = pd.DataFrame(pred_output,columns=['RecoE'])
df_final = pd.concat([df1,df2],axis=1)


assert(df1.shape[0]==len(pred_output))
assert(df_final.shape[0]==df2.shape[0])

df_final.to_csv("ml_results.csv", float_format = '%.3f')








"""
ns_probs = [0 for _ in range(len(mc_test))]

model = LogisticRegression(solver='lbfgs')
model.fit(reco_train, mc_train)

probs = model.predict_proba(reco_test)
probs_pos = pred_output[:, 1]

ns_auc = roc_auc_score(mc_test, ns_probs)
lr_auc = roc_auc_score(mc_test, probs_pos)

print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

ns_fpr, ns_tpr, _ = roc_curve(mc_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(mc_test, lr_probs)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

"""


#print(reco_train)
#print(reco_train.shape)
#print('----------------------------------------------------------------------------')
#print(reco_test)
#print(reco_test.shape)




# assert(dfsel.isnull().any().any()==False) for checking nan values (from Lilia's code)









    



    

    

    




    

        

