import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import simulate_gains

df = pd.read_csv('AllGoldDataDataPrice.csv', parse_dates=['Date'], index_col=['Date'])
df.dropna(inplace=True)

threshold = 0
window_length = 10

# if the next day is greater, set BUY to 1
df['BUY'] = (df['USD'] - df['USD'].shift() > threshold).shift(-1).fillna(0).astype(int)

df['PERCENT_USD'] = ((df['USD'] - df['USD'].shift()) / df['USD'].shift())
df['PERCENT_VIX'] = ((df['VIX'] - df['VIX'].shift()) / df['VIX'].shift())

#df.plot(y='USD', use_index=True)
#print(simulate_gains.simulate_gains(df.loc[:, 'USD'], np.ones(len(df.loc[:, 'BUY']))))
#plt.show()

# Get rolling windows
for i in range(0, window_length):
    df['USD_N-' + str(i)] = df['PERCENT_USD'].shift(i)
for i in range(0, window_length):
    df['VIX_N-' + str(i)] = df['PERCENT_VIX'].shift(i)

df['N+1'] = df['PERCENT_USD'].shift(-1)

# Drop days where there is not enough data to view past N days
df.dropna(inplace=True)

# transformer = preprocessing.StandardScaler().fit(df.loc[:, 'N-0':('N-'+str(window_length-1))])
# transformed = transformer.transform(df.loc[:, 'N-0':('N-'+str(window_length-1))])
#
# for i in range(0, window_length):
#      df['N-' + str(i)] = transformed[:,i]



features = df.loc[:, 'USD_N-0':('VIX_N-' + str(window_length - 1))]
correct_vals = df.loc[:, "N+1"]


X_train, X_test, y_train, y_test = train_test_split(features, correct_vals, test_size=0.2, shuffle=False)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

regr = LinearRegression().fit(X_train, y_train)
y_pred = regr.predict(X_test)
#np.set_printoptions(threshold=sys.maxsize)
# y_pred = np.where(clf.predict_proba(X_test)[:,0] > 0.6, 0, 1)
#print("y pred")
#print(y_pred)

prices_train, prices_test, buy_train, buy_test = train_test_split(df.loc[:, 'USD'], df.loc[:, 'BUY'], test_size=0.2, shuffle=False)
predictiondf = pd.DataFrame(prices_test)
predictiondf["PREDICTED_VALS"] = y_pred
print(predictiondf)
#testdf['BUY_PREDICT'] = (testdf['PREDICTED_VALS'] - testdf['PREDICTED_VALS'].shift() > threshold).shift(-1).fillna(0).astype(int)
predictiondf['BUY_PREDICT'] = (predictiondf['PREDICTED_VALS'] > -0.0003).astype(int)
predictiondf['EXPECTED_BUY'] = buy_test

#
bot_values = simulate_gains.simulate_gains(prices_test, predictiondf['BUY_PREDICT'])
all_hold_values = simulate_gains.simulate_gains(prices_test, np.ones(len(predictiondf["PREDICTED_VALS"])))
testdf = pd.DataFrame(prices_test)
testdf["VALUE"] = bot_values
testdf["ALLHOLD"] = all_hold_values
# #print(testdf)
print("End result for bot:", bot_values[-1])
print("End result for all hold:", all_hold_values[-1])
# # print(prices_test)
# #y_pred = np.ones(2342)

#print(testdf.loc[testdf['BUY_PREDICT'] != testdf['EXPECTED_BUY']].to_string())
#print(testdf.to_string())

#prices_test.plot(y="VALUE", use_index=True)
testdf.loc[:, "VALUE":"ALLHOLD"].plot(use_index=True)

plt.show()

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (buy_test != predictiondf['BUY_PREDICT']).sum()))

print("Perror:", (buy_test != predictiondf['BUY_PREDICT']).sum() / X_test.shape[0])
#print(df.to_string())

