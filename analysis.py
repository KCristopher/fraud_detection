import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./data/Synthetic_Financial_datasets_log.csv')
print('\n', df.head(), '\n')
print('\n', df.info(show_counts=True), '\n')
print('\n',"There is " + str(df.isna().sum().sum()) + " missing values in the dataset.", '\n')
print('\n', df.describe().loc[['min', 'max', 'mean', '50%']].round(2), '\n')