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
# print('\n', "The shape of the features is: " + str(X.shape) + " and the shape of the target is: " + str(y.shape), '\n')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print('\n', "The shape of the training features is: " + str(X_train.shape) + " and the shape of the training target is: " + str(y_train.shape), '\n')
# print('\n', "The shape of the testing features is: " + str(X_test.shape) + " and the shape of the testing target is: " + str(y_test.shape), '\n')
# print('\n',X.shape[0] == X_train.shape[0] + X_test.shape[0], '\n')
trainset = pd.concat([X_train, y_train], axis=1)
testset = pd.concat([X_test, y_test], axis=1)
print('\n', 'There are {} million rows in the training data.'.format(round(X_train.shape[0]/1000000, 2)), '\n')
normalized_counts_train = trainset['isFraud'].value_counts(dropna=False, normalize=True)
# a little more than 1 in a 1000 transactions in the training data are fraudulent
print('\n', 'The normalized distribution of the target variable in the training data is: \n', '\n', normalized_counts_train, '\n')
normalized_counts_test = testset['isFraud'].value_counts(dropna=False, normalize=True)
print('\n', 'The normalized distribution of the target variable in the testing data is: \n', '\n', normalized_counts_test, '\n')
eda_sample = (
    trainset
    .groupby("isFraud", group_keys=False)
    .sample(frac=0.10, random_state=123)
)
# print('\n', trainset.head(), '\n')
# print('\n', trainset['isFraud'].value_counts(dropna = False), '\n')
print('\n', eda_sample.head(), '\n')
print('\n', eda_sample['isFraud'].value_counts(dropna = False), '\n')
