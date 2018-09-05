import pandas as pd
from sklearn import preprocessing, svm #cross_validation,
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
#import quandl
import math

dfTemp = pd.read_json('quandlData.json')
df = pd.DataFrame(dfTemp['dataset']['data'], columns=dfTemp['dataset']['column_names'])
df['HL_PCT'] = (df['High'] - df['Last'])/ df['Last'] * 100.0
df['PCT_Change'] = (df['Last'] - df['Previous Day Price']) / df['Previous Day Price'] * 100.0

df = df[['Last', 'HL_PCT', 'PCT_Change', 'Volume']]
df.fillna(-99999, inplace=True)
forecast_col = 'Last'
forecast_out = int(math.ceil(0.1 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

accurarcy = clf.score(X_test, y_test)
print(accurarcy)

df1 = pd.DataFrame()
df1['old'] = df['Last']
df1['new'] = df['Last'].shift(-2)
math.ceil(0.01 * len(df))
