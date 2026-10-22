import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from utils import get_sample_vals_from_each_col


df = pd.read_csv('./data/Synthetic_Financial_datasets_log.csv')
# print('\n', df.head(), '\n')
# print('\n', df.info(show_counts=True), '\n')
# print('\n',"There is " + str(df.isna().sum().sum()) + " missing values in the dataset.", '\n')
# print('\n', df.describe().loc[['min', 'max', 'mean', '50%']].round(2), '\n')
# print('\n', get_sample_vals_from_each_col(12, df), '\n')
# print('\n', df[df.duplicated(keep=False, subset = ['nameOrig', 'step', 'type', 'amount'])].sort_values(by=['nameOrig', 'step', 'type', 'amount']), '\n')
X = df.drop(columns=['isFraud'])
y =  df['isFraud']
print('\n', "The shape of the features is: " + str(X.shape) + " and the shape of the target is: " + str(y.shape), '\n')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('\n', "The shape of the training features is: " + str(X_train.shape) + " and the shape of the training target is: " + str(y_train.shape), '\n')
print('\n', "The shape of the testing features is: " + str(X_test.shape) + " and the shape of the testing target is: " + str(y_test.shape), '\n')
print('\n',X.shape[0] == X_train.shape[0] + X_test.shape[0], '\n')