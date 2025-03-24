#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn import model_selection
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
import warnings
warnings.simplefilter('ignore')

# Load the training dataset
train_df = pd.read_csv('ant-1.3.csv')

# Load the testing dataset
test_df = pd.read_csv('ant-1.7.csv')

# Input features for training
X_train = train_df.iloc[:, 0:19]

# Output variable for training
y_train = train_df.iloc[:, 20]

# Input features for testing
X_test = test_df.iloc[:, 0:19]

# Output variable for testing
y_test = test_df.iloc[:, 20]

# Standardizing the data
sc = StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)

# Update the base classifiers
base_classifiers = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', XGBClassifier(random_state=42)),
    ('ada', AdaBoostClassifier(random_state=42))
]

# Update the stacking classifier
stacking_classifier = StackingCVClassifier(
    classifiers=[clf for _, clf in base_classifiers],
    meta_classifier=XGBClassifier(random_state=42),
    random_state=42
)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Ensure the same scaler is used
    ('stacking', stacking_classifier)
])

# Update the parameter grid for RandomizedSearchCV
param_grid = {
    'stacking__randomforestclassifier__n_estimators': [10, 50, 100],
    'stacking__randomforestclassifier__max_depth': [5, 10, 20],
    'stacking__xgbclassifier__n_estimators': [100, 200, 300],
    'stacking__xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
    'stacking__xgbclassifier__max_depth': [3, 4, 5],
    'stacking__xgbclassifier__min_child_weight': [1, 2, 3],
    'stacking__xgbclassifier__gamma': [0, 0.1, 0.2],
    'stacking__xgbclassifier__colsample_bytree': [0.8, 0.9, 1.0],
    'stacking__adaboostclassifier__n_estimators': [50, 100, 150],
    'stacking__adaboostclassifier__learning_rate': [0.01, 0.1, 0.2],
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, scoring='accuracy', n_iter=20, cv=5, verbose=2)
random_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = random_search.best_params_
best_accuracy = random_search.best_score_
final_model = random_search.best_estimator_

# Evaluate the final model on the test set
y_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Display results
print(f'Best hyperparameters: {best_params}')
print(f'Best cross-validated accuracy: {best_accuracy:.2f}')
print(f'Accuracy on the test set: {test_accuracy:.2f}')

# Display classification report
print(classification_report(y_test, y_pred))

# Additional metrics
f1 = f1_score(y_test, y_pred)
print('F1-Score:', f1)

precision = precision_score(y_test, y_pred, average='macro')
print('Precision:', precision)

recall_positive = recall_score(y_test, y_pred, pos_label=1)
recall_negative = recall_score(y_test, y_pred, pos_label=0)
print('Recall for positive class:', recall_positive)
print('Recall for negative class:', recall_negative)

auc = roc_auc_score(y_test, y_pred)
print('AUC:', auc)

# Matthews Correlation Coefficient
MCC_MC1 = matthews_corrcoef(y_test, y_pred)
print('Matthews Correlation Coefficient:', MCC_MC1)

print('Stacking model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

