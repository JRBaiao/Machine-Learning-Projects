# Linear Regression
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

carseats = pd.read_csv('Carseats.csv')
# print(carseats) # testing the file. 

# Separeting the dependent variable from the rest. 
y = carseats['Sales']
X = carseats.drop('Sales', axis=1)

# Splting the data into train and test.
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Transforming categorical varibales into dummy variables.
train_X_enc = pd.get_dummies(train_X, drop_first=True, dtype=int)
test_X_enc = pd.get_dummies(train_X, drop_first=True, dtype=int)
# print(train_X_enc.head())

# Now, run the model of OLS.
lm = sm.OLS(endog = train_y, # endog is dependent variable.
            exog = sm.add_constant(train_X_enc)).fit()
print(lm.summary())

