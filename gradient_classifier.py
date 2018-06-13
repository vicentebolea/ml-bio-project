import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from vars_local import *
import dataset

# Read column names from file
df = pd.read_csv(TRAIN_PATH)

df = dataset.prepare_features(df)

X = df
X = X.drop('Cancer_Type', axis=1)

y=df['Cancer_Type']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)

print("GradientBoosting")
clf_rf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, verbose=True)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

acc_score = accuracy_score(y_test, y_pred_rf)
print(precision_score(y_test, y_pred_rf, average='weighted'))
print(precision_score(y_test, y_pred_rf, average=None))

print(acc_score)
print(y_pred_rf)
