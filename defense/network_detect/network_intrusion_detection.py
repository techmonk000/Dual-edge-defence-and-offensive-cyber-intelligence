import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabulate import tabulate
import optuna

train = pd.read_csv('/content/Train_data.csv')
test = pd.read_csv('/content/Test_data.csv')

print(f"Number of duplicate rows: {train.duplicated().sum()}")
sns.countplot(x=train['class'])
print('Class distribution Training set:')
print(train['class'].value_counts())

def le(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])
le(train)
le(test)

train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

X_train = train.drop(['class'], axis=1)
Y_train = train['class']
rfc = RandomForestClassifier()
from sklearn.feature_selection import RFE
rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)
selected_features = [v for i, v in zip(rfe.get_support(), X_train.columns) if i]
X_train = X_train[selected_features]

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.transform(test)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)

models = {
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTreeClassifier": DecisionTreeClassifier(criterion="entropy", max_depth=4),
    "RandomForestClassifier": RandomForestClassifier(),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "LGBMClassifier": LGBMClassifier()
}

optuna.logging.set_verbosity(optuna.logging.WARNING)

for name, model in models.items():
    model.fit(x_train, y_train)

preds = {name: models[name].predict(x_test) for name in models}
f1_scores = {name: f1_score(y_test, preds[name], average="weighted") for name in preds}

best_model_name = max(f1_scores, key=f1_scores.get)
best_model = models[best_model_name]

data = [[name, f1_scores[name]] for name in f1_scores]
col_names = ["Model", "F1-Score"]
print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))

print(f"Best Model: {best_model_name}")
print(f"F1-Score: {f1_scores[best_model_name]}")

best_model.fit(x_train, y_train)
final_predictions = best_model.predict(x_test)

print("Final Model Results")
print(confusion_matrix(y_test, final_predictions))
print(classification_report(y_test, final_predictions))
