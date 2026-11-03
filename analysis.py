import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from utils import get_sample_vals_from_each_col
from sklearn.preprocessing import OneHotEncoder


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
# normalized_counts_train = trainset['isFraud'].value_counts(dropna=False, normalize=True)
# a little more than 1 in a 1000 transactions in the training data are fraudulent
# print('\n', 'The normalized distribution of the target variable in the training data is: \n', '\n', normalized_counts_train, '\n')
# normalized_counts_test = testset['isFraud'].value_counts(dropna=False, normalize=True)
# print('\n', 'The normalized distribution of the target variable in the testing data is: \n', '\n', normalized_counts_test, '\n')
# eda_sample = (
#     trainset
#     .groupby("isFraud", group_keys=False)
#     .sample(frac=0.10, random_state=123)
# )
# print('\n', trainset.head(), '\n')
# print('\n', trainset['isFraud'].value_counts(dropna = False), '\n')
# print('\n', eda_sample.head(), '\n')
# print('\n', eda_sample['isFraud'].value_counts(dropna = False), '\n')
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
# eda_sample_num = eda_sample.select_dtypes(include=['int64', 'float64'])
# print('\n', eda_sample_num.head(), '\n')

# print('\n', eda_sample_num.groupby('isFlaggedFraud')['isFraud'].value_counts(), '\n')
# print('\n', eda_sample_num.groupby('isFlaggedFraud')['isFraud'].value_counts(normalize=True), '\n')
# # isFlaggedFraud is almost always the same as isFraud, so it is not a useful feature for the model, because it's basically a duplicate of the target variable
# sns.kdeplot(data=eda_sample_num, x='amount', hue='isFraud', common_norm=False)
# plt.title('Distribution of Transaction Amounts by Fraud Status')
# plt.xlabel('Transaction Amount')
# plt.ylabel('Density')
# plt.xlim(0, 11*10**6)
# plt.show()

# plt.figure(figsize=(14, 6.5))
# sns.kdeplot(data=eda_sample_num, x='step', hue='isFraud', common_norm=False)
# plt.title('Distribution of Transaction Steps by Fraud Status')
# plt.xlabel('Transaction Step')
# plt.ylabel('Density')
# plt.xlim(0, 750)
# plt.xticks(np.arange(0, 800, 50))
# plt.grid()
# plt.show()

# eda_sample_plotting = eda_sample_num.sample(frac=0.1, random_state=123)
# print('\n', eda_sample_plotting['isFraud'].value_counts(normalize=True), '\n')
# print('\n', eda_sample_plotting['isFraud'].value_counts(normalize=False), '\n')


# for c in ['step','amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
#     sns.boxenplot(data=eda_sample, x='isFraud', y=c)
#     plt.title(f'Distribution of {c} by Fraud Status')
#     plt.show()


"""boxenplot "Distribution of step by Fraud Status" compares the distribution of step for non-fraudulent (0) and fraudulent (1) transactions.
- Fraudulent transactions occur later overall: their median step is around 355, compared with 240 for legitimate transactions.
- Fraud is spread across a wider time range: the middle 50% of fraudulent transactions spans approximately 175–550, versus 155–330 for legitimate transactions.
- There is substantial overlap, so step alone would not clearly separate fraud from legitimate transactions."""

""""Distribution of amount by Fraud Status" reveals that fraudulent transactions generally involve larger amounts, but the largest individual transactions are legitimate.
- Legitimate transactions (0): most amounts are concentrated near zero on this scale, with a long right tail reaching roughly 55 million.
- Fraudulent transactions (1): the central boxes sit higher and span a wider range, indicating generally larger and more variable amounts.
  The upper tail reaches approximately 10 million.
- Both distributions are strongly right-skewed: a small proportion of transactions have amounts much larger than typical ones.
The legitimate extremes compress the plot, making the medians and overlap hard to assess."""

"""Distribution of oldbalanceOrg by Fraud Status" shows that fraudulent transactions tend to come from accounts with higher starting balances (oldbalanceOrg).
- Legitimate transactions (0): the central 50% is tightly concentrated near zero on this scale.
- Fraudulent transactions (1): the central 50% spans a visibly wider and higher balance range.
- Both distributions have long right tails, with extreme balances reaching roughly 37 million in both groups.
A very high starting balance therefore does not uniquely indicate fraud."""

"""Distribution of newbalanceOrig by Fraud Status" shows that fraudulent transactions have their resulting sender balance (newbalanceOrig) much more concentrated near zero.
- Fraud (1): most boxes collapse near zero, with a small upper tail and extreme values reaching about 27 million.
- Legitimate (0): typical balances are also low on this scale, but the upper quantiles extend much higher, with extremes around 37 million."""

"""Distribution of oldbalanceDest by Fraud Status" shows that fraudulent transactions have a less extended upper tail in the recipient’s initial balance (oldbalanceDest).
- Both classes are concentrated near zero on this scale, so their medians and central ranges are difficult to compare.
- Legitimate transactions (0) have a much longer upper tail, reaching approximately 300 million.
- Fraudulent transactions (1) reach approximately 30 million, with upper boxes also closer to zero."""

"""Distribution of newbalanceDest by Fraud Status" reveals that the recipient’s final balance (newbalanceDest) has a much longer upper tail for legitimate transactions.
- Legitimate (0): extreme balances reach approximately 320 million.
- Fraud (1): extremes reach approximately 30 million.
- Both groups’ central boxes are compressed near zero, making typical balances and their differences difficult to assess."""

# Transactions identified as fraudulent are annulled. 
# Hence, for fraud detection analysis, the following columns should not be utilized: oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest.
trainset_selected = trainset.drop(columns=['isFlaggedFraud', 'nameOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig'])
testset_selected = testset.drop(columns=['isFlaggedFraud', 'nameOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrg', 'newbalanceOrig'])

X_train_sel, y_train = trainset_selected.drop(columns=['isFraud']), trainset_selected['isFraud']
X_test_sel, y_test = testset_selected.drop(columns=['isFraud']), testset_selected['isFraud']

print('\n', f"There are {X_train_sel.shape[0] / 10 ** 6:.3f} million rows and {X_train_sel.shape[1]} columns in selected training data.", '\n')
print('\n', f"There are {X_test_sel.shape[0] / 10 ** 6:.3f} million rows and {X_test_sel.shape[1]} columns in selected test data.", '\n')
print('\n', trainset_selected.head(), '\n')

# since fraud happens only in CASH_OUT and TRANSFER transactions, we can group the other types of transactions into a single category called OTHER
X_train_sel['type'] = X_train_sel['type'].map({'CASH_OUT': 'CASH_OUT', 'PAYMENT': 'OTHER', 'CASH_IN': 'OTHER', 'TRANSFER': 'TRANSFER', 'DEBIT': 'OTHER'})
X_test_sel['type'] = X_test_sel['type'].map({'CASH_OUT': 'CASH_OUT', 'PAYMENT': 'OTHER', 'CASH_IN': 'OTHER', 'TRANSFER': 'TRANSFER', 'DEBIT': 'OTHER'})

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categoricals = ohe.fit_transform(X_train_sel[['type']])
categoricals_test = ohe.transform(X_test_sel[['type']])
X_train_sel_enc = pd.concat([X_train_sel.drop(columns=['type']).reset_index(drop=True), pd.DataFrame(categoricals, columns=ohe.get_feature_names_out(['type']))], axis=1)
X_test_sel_enc = pd.concat([X_test_sel.drop(columns=['type']).reset_index(drop=True), pd.DataFrame(categoricals_test, columns=ohe.get_feature_names_out(['type']))], axis=1)

print('\n', X_train_sel_enc.head(), '\n')