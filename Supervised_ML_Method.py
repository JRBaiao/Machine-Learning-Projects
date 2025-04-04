# Supervised Machine Learning method - Linear Regression.
import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Toy dataset.
diabetes = datasets.load_diabetes()
# print(diabetes.feature_names)

df = pd.DataFrame(diabetes.data)
df.columns = ['age', 'sex', 
              'bmi', 'bp', 
              's1', 's2', 
              's3', 's4', 
              's5', 's6']
df['target'] = diabetes.target
# print(df.head())

# StatsModels Package
lm = sm.OLS(endog = df['target'], # endog is dependent variable.
            exog = sm.add_constant(df[df.columns[0:10]])).fit()
# print(lm.summary())

""" In ML, regeression is used mainly for prediction. The usual 
practice is to split the data in train and test data: the model will be fit 
to the train data and the test data will be used for prediction. """

X = diabetes.data # set of idependet variables
y = diabetes.target # my dependet variable

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Next step, run the model on the train data and test/predict on the test data. 
lr = LinearRegression()
lr.fit(train_X, train_y)
pred_y = lr.predict(test_X)

# Output of actual vs. predicted values.
result = pd.DataFrame({'Actual': test_y, 'Predicted': pred_y}) 
print(result)

print('---')

# R-Squared calculation.
print(r2_score(test_y, pred_y)) # the closer to 1, the better your R-Squared is. 
