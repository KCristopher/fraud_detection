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
# print('\n', 'There are {} million rows in the training data.'.format(round(X_train.shape[0]/1000000, 2)), '\n')
normalized_counts_train = trainset['isFraud'].value_counts(dropna=False, normalize=True)
# a little more than 1 in a 1000 transactions in the training data are fraudulent
# print('\n', 'The normalized distribution of the target variable in the training data is: \n', '\n', normalized_counts_train, '\n')
normalized_counts_test = testset['isFraud'].value_counts(dropna=False, normalize=True)
# print('\n', 'The normalized distribution of the target variable in the testing data is: \n', '\n', normalized_counts_test, '\n')
eda_sample = (
    trainset
    .groupby("isFraud", group_keys=False)
    .sample(frac=0.10, random_state=123)
)
# print('\n', trainset.head(), '\n')
# print('\n', trainset['isFraud'].value_counts(dropna = False), '\n')
print('\n', eda_sample.head(), '\n')
print('\n', eda_sample['isFraud'].value_counts(dropna = False), '\n')
# eda_sample_cat = eda_sample.select_dtypes(include=['object'])
# eda_sample_cat = pd.concat([eda_sample_cat, eda_sample['isFraud']], axis=1)
# print('\n', "The number of unique values in each categorical column is: \n", '\n', eda_sample_cat.drop(columns=['isFraud']).nunique(), '\n')
# only type have the number of unique values small enough for grouping and plotting
# besides, nameOrig and nameDest can be useful only for customer-level analysis, which is not the focus of this project
# eda_sample_cat.groupby('type')['isFraud'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=False, figsize=(7.5, 7.5))
# plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions by Type')
# plt.xlabel('Transaction Type')
# plt.ylabel('Proportion of Transactions')
# plt.ylim(0, 0.025)
# plt.yticks(np.arange(0, 0.025, 0.0025 / 2))
# plt.grid(axis='y', linestyle='--')
# plt.show()
# eda_sample_cat.groupby('isFraud')['type'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, figsize=(7.5, 7.5))
# plt.title('Fraudulent and Non-Fraudulent Transactions by Type')
# plt.xlabel('Non-Fraudulent (0) compared to Fraudulent (1)')
# plt.ylabel('Proportion of Transactions')
# plt.show()
# eda_sample_cat['type'].value_counts(normalize=True).plot(kind='pie', figsize=(7.5, 7.5), autopct='%1.1f%%', startangle=90, counterclock=False)
# plt.title('Distribution of Transaction Types')
# plt.ylabel('')
# plt.show()
eda_sample_num = eda_sample.select_dtypes(include=['int64', 'float64'])
print('\n', eda_sample_num.head(), '\n')

print('\n', eda_sample_num.groupby('isFlaggedFraud')['isFraud'].value_counts(), '\n')
print('\n', eda_sample_num.groupby('isFlaggedFraud')['isFraud'].value_counts(normalize=True), '\n')
# isFlaggedFraud is almost always the same as isFraud, so it is not a useful feature for the model, because it's basically a duplicate of the target variable
# sns.kdeplot(data=eda_sample_num, x='amount', hue='isFraud', common_norm=False)
# plt.title('Distribution of Transaction Amounts by Fraud Status')
# plt.xlabel('Transaction Amount')
# plt.ylabel('Density')
# plt.xlim(0, 11*10**6)
# plt.show()

plt.figure(figsize=(14, 6.5))
sns.kdeplot(data=eda_sample_num, x='step', hue='isFraud', common_norm=False)
plt.title('Distribution of Transaction Steps by Fraud Status')
plt.xlabel('Transaction Step')
plt.ylabel('Density')
plt.xlim(0, 750)
plt.xticks(np.arange(0, 800, 50))
plt.grid()
plt.show()