__author__ = 'emarrig'

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('/home/marik0/repos/kaggle/higgs/data/training.csv', index_col='EventId')
X_test = pd.read_csv('/home/marik0/repos/kaggle/higgs/data/test.csv', index_col='EventId')

y = df['Label']
weights = df['Weight']
X = df.drop(['Label', 'Weight'], axis=1)

clf = GradientBoostingClassifier(n_estimators=80,
                                 max_depth=10,
                                 min_samples_leaf=200,
                                 max_features=10,
                                 verbose=1).fit(X, y)

p = clf.predict_proba(X_test)[:, 1]

# Create the rank order from the predicted probabilities
rank_order = np.argsort(p) + 1

# Create the class column
cl = np.empty(len(X_test), dtype=np.object)
cl[p > 0.8] = 's'
cl[p <= 0.8] = 'b'

# Get the event Id from the initial data frame
ids = X_test.index.values

# Create a data frame with the results and save everything to a CSV file
result_df = pd.DataFrame({"EventId": ids, "RankOrder": rank_order, "Class": cl})
result_df.to_csv("result.csv", index=False, columns=["EventId", "RankOrder", "Class"])