import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

import simulate_gains

df = pd.read_csv('AllGoldVixData.csv', parse_dates=['Date'], index_col=['Date'])

df.dropna(inplace=True)

threshold = 0
window_length = 10
k = 10

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

# Drop days where there is not enough data to view past N days
df.dropna(inplace=True)

# df = df[[c for c in df if c not in ['VIX']]
#        + ['VIX']]

transformer = preprocessing.StandardScaler().fit(df.loc[:, 'USD_N-0':'VIX_N-'+str(window_length-1)])
transformed = transformer.transform(df.loc[:, 'USD_N-0':'VIX_N-'+str(window_length-1)])

#df['VIX'] = transformed[:,window_length]


kf = KFold(n_splits=k)

features = df.loc[:, 'USD_N-0':('VIX_N-' + str(window_length - 1))]
labels = df.loc[:, "BUY"]

prob_thresholds = np.linspace(0.5, 0.6, num=21)
#prob_thresholds = np.array([0.58]) # best for k = 10 is 0.594

best_gain_percentage = float('-inf')
best_gain_percentage_thresh = -1
best_Perror = 1
best_Perror_thresh = -1

for prob_thresh in prob_thresholds:
    thresh_out_of_range = False

    Perrors = np.zeros(k)
    gain_percentages = np.zeros(k)
    for i, (train_index, test_index) in enumerate(kf.split(features, labels)):
        X_train , X_test = features.iloc[train_index,:],features.iloc[test_index,:]
        y_train , y_test = labels[train_index] , labels[test_index]
        # print(X_train)
        # print(X_test)
        # print(y_train)
        # print(y_test)

        clf = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
        #y_pred = clf.predict(X_test)
        #np.set_printoptions(threshold=sys.maxsize)
        y_pred = np.where(clf.predict_proba(X_test)[:,0] > prob_thresh, 0, 1)
        #print("y pred")
        #print(y_pred)

        prices_test = df.loc[:, 'USD'][test_index]

        bot_values = simulate_gains.simulate_gains(prices_test, y_pred)
        all_hold_values = simulate_gains.simulate_gains(prices_test, np.ones(len(y_pred)))
        testdf = pd.DataFrame(prices_test)
        testdf["VALUE"] = bot_values
        testdf["ALLHOLD"] = all_hold_values
        #print(testdf)
        print("End result for bot:", bot_values[-1])
        print("End result for all hold:", all_hold_values[-1])
        # print(prices_test)
        #y_pred = np.ones(2342)

        if bot_values[-1] == all_hold_values[-1]:
            thresh_out_of_range = True
        gain_percentages[i] = ((bot_values[-1] - all_hold_values[-1]) / all_hold_values[-1])
        Perrors[i] = (y_test != y_pred).sum() / X_test.shape[0]
        print("NUm days", X_test.shape[0])

        testdf.loc[:, "VALUE":"ALLHOLD"].plot(use_index=True)
        plt.ylabel("Value ($)")
        plt.title("Returns of holding gold vs. using LDA model\nwith threshold 0.52 and k = 5 folds")
        plt.legend(["Portfolio value using active model", "Portfolio value passively \nholding gold for whole period"])

        plt.show()

        # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    avg_gain_percentage = np.average(gain_percentages)
    avg_Perror = np.average(Perrors)
    if not thresh_out_of_range:
        if avg_gain_percentage > best_gain_percentage:
            best_gain_percentage = avg_gain_percentage
            best_gain_percentage_thresh = prob_thresh
        if avg_Perror < best_Perror:
            best_Perror = avg_Perror
            best_Perror_thresh = prob_thresh


print("Best average daily gain percentage:", best_gain_percentage)
print("Best gain percentage thresh:", best_gain_percentage_thresh)
print("Best Perror:", best_Perror)
print("Best Perror thresh:", best_Perror_thresh)
    #print(df.to_string())

