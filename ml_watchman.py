"""
A script using the machine learning framework GradientBoostingRegressor from sklearn to more accuratly reconstruct the energy of events within a simulated WATCHMAN detector.

Author: L. Dorman-Gajic
"""
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.stats import norm

from sklearn import svm, datasets
from sklearn import model_selection
from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_curve, auc, plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance


if len(sys.argv) != 4 :
    print("Wrong Number of Arguments")
    print("Usage: " + sys.argv[0] + "<true data input file>" + "<reconstructed data input file>" + "<output file>")
else :
    #opening files in command line, one for true parameters and one for the reconstructed
    mc_infile = sys.argv[1]
    reco_infile = sys.argv[2]
    outfile_name = sys.argv[3]

#random seed to improve reproducibility
seed = 150
np.random.seed(seed)

#reading files into data frame
mc = open(str(mc_infile))
print("true data: ",mc)
mc_read = pd.read_csv(mc)
df_mc = mc_read[['gtid', 'mcx', 'mcy', 'mcz', 'mc_energy', 'true_wall_r',  'true_wall_z']]

reco = open(str(reco_infile))
reco_read = pd.read_csv(reco)
df_reco = reco_read[['gtid', 'x', 'y', 'z', 'pe', 'closestPMT', 'n100', 'reco_wall_r', 'reco_wall_z']]

#print(df_reco)

#normalising variables in files
df_reco_n = pd.DataFrame([df_reco['x']/10000, df_reco['y']/10000, df_reco['z']/10000, df_reco['pe']/400, df_reco['closestPMT']/7000, df_reco['n100']/300, df_reco['reco_wall_r']/10000, df_reco['reco_wall_z']/10000]).T

print("check normalisation of reconstructed data:")
print(df_reco_n.head(), df_reco_n.shape)
print("----------------------------------------------------------------------------")


df_mc_n = pd.DataFrame([df_mc['mcx']/10000, df_mc['mcy']/10000, df_mc['mcz']/10000,  df_mc['mc_energy']/10, df_mc['true_wall_r']/10000, df_mc['true_wall_z']/10000]).T

print("check normalisation of true data: ")
print(df_mc_n.head(), df_mc_n.shape)
print("----------------------------------------------------------------------------")


mc_array = np.array(df_mc_n[['mc_energy']])

reco_array = np.array(df_reco_n[['x', 'y', 'z', 'pe', 'closestPMT', 'n100', 'reco_wall_r', 'reco_wall_z']])

rnd_indices = np.random.rand(len(reco_array)) < 0.50

mc_array_train = mc_array[rnd_indices]
reco_array_train = reco_array[rnd_indices]

mc_array_pred = mc_array[~rnd_indices]
reco_array_pred = reco_array[~rnd_indices]

print('number of training events and shape(true): ', len(mc_array_train), ' ', mc_array_train.shape,  ' number of predicting events and shape: ', len(mc_array_pred), ' ', mc_array_pred.shape)


print('number of training events and shape(reco): ', len(reco_array_train), ' ', reco_array_train.shape,  ' number of predicting events and shape: ', len(reco_array_pred), ' ', reco_array_pred.shape)


print('----------------------------------------------------------------------------')


n_estimators=500
params = {'n_estimators':n_estimators, 'max_depth': 10,
    'learning_rate': 0.01, 'loss': 'lad'}

offset = int(reco_array_train.shape[0]*0.7)
reco_train, mc_train = reco_array_train[:offset], mc_array_train[:offset].reshape(-1)
reco_test, mc_test = reco_array_train[offset:], mc_array_train[offset:].reshape(-1)

print("train shape reco: ", reco_train.shape, "train shape mc: ", mc_train.shape)
print("test shape reco: ", reco_test.shape, "test shape mc: ", mc_test.shape)

gbr = ensemble.GradientBoostingRegressor(**params)
gbr.fit(reco_train, mc_train)

print("Feature Importances")
print(gbr.feature_importances_)

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

df_final.to_csv(outfile_name, float_format = '%.3f')


fig = plt.figure()
plt.subplot(1,1,1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, gbr.train_score_, 'b-', label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()



features = np.array(('x', 'y', 'z', 'pe', 'closestPMT', 'n100', 'reco_wall_r', 'reco_wall_z'))
print(features)
feature_importance = gbr.feature_importances_
print(feature_importance)
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, features[sorted_idx])
plt.title('Feature Importance (MDI)')
print(features[sorted_idx])

result = permutation_importance(gbr, reco_test, mc_test, n_repeats=10,
                                random_state=42, n_jobs=2)
print(result)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=features[sorted_idx])
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()

"""
    
infile = 'wbls_merged_130000_ml_output_500.csv'
file = open(str(infile))
file_read = pd.read_csv(file)
df = file_read[['TrueEnergy', 'RecoE']]
file_array = np.array(df[['TrueEnergy', 'RecoE']])
TrueEnergy = file_array.T[0]
RecoE = file_array.T[1]

data = []
for i in range(len(RecoE)):
    point = (TrueEnergy[i] - RecoE[i])/TrueEnergy[i]
    if point > 0.3 or point < -0.3:
        continue
    else:
        data.append(point)


mu, std = norm.fit(data)
FWHM = 2*np.sqrt(2*np.log(2))*std
plt.hist(data, bins=100, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f, fwhm = %.2f" % (mu, std, FWHM)
plt.title(title)
plt.xlabel('(true energy - reconstructed energy)/true energy')
plt.show()


eval_input_fn = make_input_fn(reco_test, mc_test, shuffle=False, n_epochs=1)
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
fpr, tpr, _ = roc_curve(mc_test, probs)


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









    



    

    

    




    

        

