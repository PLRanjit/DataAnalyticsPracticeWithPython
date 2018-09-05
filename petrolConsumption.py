import pandas as pd
import numpy as np
dataset = pd.read_csv('Data/petrol_consumption.csv')

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

#print(X)
#print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#regressor = RandomForestRegressor(n_estimators=20, random_state=0)
#regressor.fit(X_train, y_train)
#y_pred = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df.describe())

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

sample = [[7.50,4870,2351,0.5290]]
preds = regressor.predict(sample)

print("Predictions:", preds) 
