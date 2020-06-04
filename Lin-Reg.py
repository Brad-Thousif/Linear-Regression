import pandas as pd
import scipy as sp
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import linalg
filepath = 'hp.csv'
df = pd.read_csv(filepath)
df.dropna(inplace=True) 
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(df.drop('Class', axis=1), df['Class'], test_size=.20, random_state=100)
regr=LinearRegression()
regr.fit(xtrain,ytrain)
ypred=regr.predict(xtest)
ypred=ypred.astype(int)

# Pred and Truth
pred_actuals = pd.DataFrame([(pred, act) for pred, act in zip(ypred, ytest)], columns=['pred', 'true'])
print(pred_actuals[:5])
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
truth = pred_actuals.loc[:, 'true']
pred = pred_actuals.loc[:, 'pred']
cm=confusion_matrix(truth, pred)
print('\nConfusion Matrix: \n', confusion_matrix(truth, pred))
print('Recall  = {}'.format((cm[0][0])/(cm[0][0]+cm[1][0])))
print('F1 Score = {}'.format(((2*cm[0][0]))/((2*cm[0][0])+cm[0][1]+cm[1][0])))
print('\nAccuracy Score: ', accuracy_score(truth, pred))
print('\nClassification Report: \n', classification_report(truth, pred))
