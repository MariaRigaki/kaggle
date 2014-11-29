__author__ = 'marik0'
import pandas as pd
import numpy as np

df_train = pd.read_csv('/home/marik0/repos/kaggle/higgs/data/training.csv', index_col='EventId')
df_test = pd.read_csv('/home/marik0/repos/kaggle/higgs/data/test.csv', index_col='EventId')

df_train = df_train.replace('-999.0', np.nan)
df_test = df_test.replace('-999.0', np.nan)

df_train = df_train.fillna(df_train.median())
df_test = df_test.fillna(df_test.median())


print df_test.head()
print df_train.head()

df_train.to_csv('new_train.csv')
df_test.to_csv('new_test.csv')


