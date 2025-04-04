import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

diabetes = pd.read_csv('diabetes.csv')
# print(diabetes.head())

# Separeting the dependent variable from the idenpendent variables. 
y = diabetes['Outcome']
X = diabetes.drop('Outcome', axis=1)
# print(y.head())

# Train and Test. 
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
# print(train_X.shape, train_y.shape, test_X.shape, train_y.shape)

# Running the Logistic Regression on the train data.
log_reg = LogisticRegression(max_iter=500)
print(log_reg.fit(train_X, train_y))

# Prediction on the test data. 
lr_pred = log_reg.predict(test_X)

# Checking the quality of the prediction on the model.
# print('Confusion Matrix: \n', confusion_matrix(test_y, lr_pred))
print('Classification Report: \n', classification_report(test_y, lr_pred))