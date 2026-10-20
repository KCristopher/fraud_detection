import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./data/Synthetic_Financial_datasets_log.csv')
print('\n', df.head(), '\n')
print('\n', df.info(), '\n')