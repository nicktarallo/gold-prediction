import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn import metrics


import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

import simulate_gains

df = pd.read_csv('AllGoldDataDataPrice.csv', parse_dates=['Date'], index_col=['Date'])
df.dropna(inplace=True)

threshold = 0
window_length = 10

# if the next day is greater, set BUY to 1
df['BUY'] = (df['USD'] - df['USD'].shift() > threshold).shift(-1).fillna(0).astype(int)

df['PERCENT'] = ((df['USD'] - df['USD'].shift()) / df['USD'].shift())

df.plot(y='USD', use_index=True)
print(simulate_gains.simulate_gains(df.loc[:, 'USD'], np.ones(len(df.loc[:, 'BUY']))))
plt.show()

# Get rolling windows
for i in range(0, window_length):
    df['N-' + str(i)] = df['PERCENT'].shift(i)

# Drop days where there is not enough data to view past N days
df.dropna(inplace=True)

transformer = preprocessing.Normalizer().fit(df.loc[:, 'N-0':('N-'+str(window_length-1))])
transformed = transformer.transform(df.loc[:, 'N-0':('N-'+str(window_length-1))])

for i in range(0, window_length):
     df['N-' + str(i)] = transformed[:,i]

features = df.loc[:, 'N-0':('N-'+str(window_length-1))]
labels = df.loc[:, "BUY"]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
#print(X_test)

clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#np.set_printoptions(threshold=sys.maxsize)
y_pred = np.where(clf.predict_proba(X_test)[:,0] > 0.6, 0, 1) #0.6
print("y pred")
print(y_pred)

prices_train, prices_test = train_test_split(df.loc[:, 'USD'], test_size=0.2, shuffle=False)

bot_values = simulate_gains.simulate_gains(prices_test, y_pred)
all_hold_values = simulate_gains.simulate_gains(prices_test, np.ones(len(y_pred)))
testdf = pd.DataFrame(prices_test)
testdf["VALUE"] = bot_values
testdf["ALLHOLD"] = all_hold_values
#print(testdf)
print("End result for bot:", bot_values[-1])
print("End result for all hold:", all_hold_values[-1])
print((bot_values[-1] - all_hold_values[-1]) / all_hold_values[-1])
# print(prices_test)
#y_pred = np.ones(2342)

#prices_test.plot(y="VALUE", use_index=True)
testdf.loc[:, "VALUE":"ALLHOLD"].plot(use_index=True)

plt.ylabel("Value ($)")
plt.title("Returns of holding gold vs. using LDA model with threshold 0.6\n using holdout method with 20% held for testing")
#plt.title("Returns of holding gold vs. using Gaussian NB model for min P(error)\n using holdout method with 20% held for testing")
plt.legend(["Portfolio value using active model", "Portfolio value passively \nholding gold for whole period"])
plt.show()
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Don't Hold", "Hold"])

cm_display.plot()
plt.xlabel("Decision made by model")
plt.ylabel("True best decision label")

plt.show()

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

#print(df.to_string())

