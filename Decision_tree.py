# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
diabetes = pd.read_csv('diabetes.csv')

# Separate the target variable and features
y = diabetes['Outcome']
X = diabetes.drop('Outcome', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% testing

# Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=5)
clf = clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Visualize the decision tree
# plt.figure(figsize=(25, 20))  # Set figure size for better visualization
# plot_tree(
#     clf,
#     feature_names=X.columns,            
#     class_names=['No', 'Yes'],          
#     filled=True                         
# plt.show()

clf_pruned = DecisionTreeClassifier(max_depth=3, random_state=5)
clf_pruned = clf_pruned.fit(X_train, y_train)
plt.figure(figsize=(15, 10))  # Set figure size for better visualization
plot_tree(
    clf_pruned,
    feature_names=X.columns,            # Feature names
    class_names=['No', 'Yes'],          # Class names (e.g., 0='No', 1='Yes')
    filled=True                         # Color the nodes based on the class
)
plt.show()