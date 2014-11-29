__author__ = 'emarrig'

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

#df = pd.read_csv('/home/marik0/repos/kaggle/higgs/data/training.csv', index_col='EventId')
#X_test = pd.read_csv('/home/marik0/repos/kaggle/higgs/data/test.csv', index_col='EventId')
df = pd.read_csv('/home/marik0/repos/kaggle/higgs/pylearn/scaled_train.csv', index_col='EventId')
X_test = pd.read_csv('/home/marik0/repos/kaggle/higgs/pylearn/scaled_test.csv', index_col='EventId')

y = df['Label']
weights = df['Weights']
X = df.drop(['Label', 'Weights'], axis=1)

print np.shape(X)

clf = GradientBoostingClassifier(n_estimators=150,
                                 max_depth=12,
                                 min_samples_leaf=240,
                                 max_features=10,
                                 verbose=1).fit(X, y)

p = clf.predict_proba(X_test)[:, 1]

# Create the rank order from the predicted probabilities
rank_order = np.argsort(p) + 1

# Create the class column
cl = np.empty(len(X_test), dtype=np.object)
cl[p > 0.9] = 's'
cl[p <= 0.9] = 'b'

# Get the event Id from the initial data frame
ids = X_test.index.values

# Create a data frame with the results and save everything to a CSV file
result_df = pd.DataFrame({"EventId": ids, "RankOrder": rank_order, "Class": cl})
result_df.to_csv("result.csv", index=False, columns=["EventId", "RankOrder", "Class"])