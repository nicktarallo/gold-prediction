import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

import matplotlib.pyplot as plt
import simulate_gains

#df = pd.read_csv('AllGoldVixData.csv', parse_dates=['Date'], index_col=['Date'])
df = pd.read_csv('AllGoldDataDataPrice.csv', parse_dates=['Date'], index_col=['Date'])
df.dropna(inplace=True)

threshold = 0
window_length = 10
k = 10

# if the next day is greater, set BUY to 1
df['BUY'] = (df['USD'] - df['USD'].shift() > threshold).shift(-1).fillna(0).astype(int)

df['PERCENT_USD'] = ((df['USD'] - df['USD'].shift()) / df['USD'].shift())

#df.plot(y='USD', use_index=True)
#print(simulate_gains.simulate_gains(df.loc[:, 'USD'], np.ones(len(df.loc[:, 'BUY']))))
#plt.show()

# Get rolling windows
for i in range(0, window_length):
    df['USD_N-' + str(i)] = df['PERCENT_USD'].shift(i)

df['N+1'] = df['PERCENT_USD'].shift(-1)

# Drop days where there is not enough data to view past N days
df.dropna(inplace=True)

# transformer = preprocessing.Normalizer().fit(df.loc[:, 'N-0':('N-'+str(window_length-1))])
# transformed = transformer.transform(df.loc[:, 'N-0':('N-'+str(window_length-1))])
#
# for i in range(0, window_length):
#      df['N-' + str(i)] = transformed[:,i]

kf = KFold(n_splits=k)

features = df.loc[:, 'USD_N-0':('USD_N-' + str(window_length - 1))]
correct_vals = df.loc[:, "N+1"]

# prob_thresholds = np.linspace(0.5, 0.6, num=21)
# #prob_thresholds = np.array([0.52]) # 0.52 k = 5, 0.515 k = 10

best_gain_percentage = float('-inf')
best_gain_percentage_thresh = -1
best_Perror = 1
best_Perror_thresh = -1

prob_thresholds = np.linspace(-0.002, 0.002, 61) # -0.002, 0.002
prob_thresholds = np.array([-0.0008])

for prob_thresh in prob_thresholds:
    thresh_out_of_range = False

    Perrors = np.zeros(k)
    gain_percentages = np.zeros(k)
    mae_values = np.zeros(k)
    mse_values = np.zeros(k)
    for i, (train_index, test_index) in enumerate(kf.split(features, correct_vals)):
        X_train , X_test = features.iloc[train_index,:],features.iloc[test_index,:]
        y_train , y_test = correct_vals[train_index] , correct_vals[test_index]
        # print(X_train)
        # print(X_test)
        # print(y_train)
        # print(y_test)

        regr = LinearRegression().fit(X_train, y_train)

        y_pred = regr.predict(X_test)

        #np.set_printoptions(threshold=sys.maxsize)
        print("y pred")
        print(y_pred)

        prices_test = df.loc[:, 'USD'][test_index]
        buy_test = df.loc[:, 'BUY'][test_index]

        predictiondf = pd.DataFrame(prices_test)
        predictiondf["PREDICTED_VALS"] = y_pred
        # testdf['BUY_PREDICT'] = (testdf['PREDICTED_VALS'] - testdf['PREDICTED_VALS'].shift() > threshold).shift(-1).fillna(0).astype(int)
        # predictiondf['BUY_PREDICT'] = (
        #             (predictiondf['PREDICTED_VALS'] - predictiondf['USD']) / predictiondf['USD'] > prob_thresh).astype(int)
        print(predictiondf)
        predictiondf['BUY_PREDICT'] = (predictiondf['PREDICTED_VALS'] > prob_thresh).astype(int)
        predictiondf['EXPECTED_BUY'] = buy_test

        bot_values = simulate_gains.simulate_gains(prices_test, y_pred)
        all_hold_values = simulate_gains.simulate_gains(prices_test, np.ones(len(y_pred)))
        testdf = pd.DataFrame(prices_test)
        testdf["VALUE"] = bot_values
        testdf["ALLHOLD"] = all_hold_values
        bot_values = simulate_gains.simulate_gains(prices_test, predictiondf['BUY_PREDICT'])
        all_hold_values = simulate_gains.simulate_gains(prices_test, np.ones(len(predictiondf["PREDICTED_VALS"])))
        testdf = pd.DataFrame(prices_test)
        testdf["VALUE"] = bot_values
        testdf["ALLHOLD"] = all_hold_values
        #print(testdf)
        #print("End result for bot:", bot_values[-1])
        #print("End result for all hold:", all_hold_values[-1])
        # print(prices_test)
        #y_pred = np.ones(2342)

        if bot_values[-1] == all_hold_values[-1]:
            #pass
            thresh_out_of_range = True
        gain_percentages[i] = (bot_values[-1] - all_hold_values[-1]) / all_hold_values[-1]
        #gain_percentages[i] = ((bot_values[-1] - all_hold_values[-1]) / all_hold_values[-1]) / X_test.shape[0]
        Perrors[i] = (buy_test != predictiondf['BUY_PREDICT']).sum() / X_test.shape[0]

        mean_absolute_error(buy_test, predictiondf['BUY_PREDICT'])
        print("MSE,", mean_squared_error(buy_test, predictiondf['BUY_PREDICT']))

        #prices_test.plot(y="VALUE", use_index=True)
        testdf.loc[:, "VALUE":"ALLHOLD"].plot(use_index=True)
        plt.ylabel("Value ($)")
        plt.title(
            "Returns of holding gold vs. using Rolling Window Linear Regression\nClassificationwith threshold = -0.0008 using K = 10 folds")
        # plt.title("Returns of holding gold vs. using Gaussian NB model for min P(error)\n using holdout method with 20% held for testing")
        plt.legend(["Portfolio value using active model", "Portfolio value passively \nholding gold for whole period"])

        plt.show()

        # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    avg_gain_percentage = np.average(gain_percentages)
    avg_Perror = np.average(Perrors)
    print(gain_percentages)
    print(Perrors)
    if not thresh_out_of_range:
        if avg_gain_percentage > best_gain_percentage:
            best_gain_percentage = avg_gain_percentage
            best_gain_percentage_thresh = prob_thresh
        if avg_Perror < best_Perror:
            best_Perror = avg_Perror
            best_Perror_thresh = prob_thresh

print("Best average gain percentage:", best_gain_percentage)
print("Best gain percentage thresh:", best_gain_percentage_thresh)
print("Best Perror:", best_Perror)
print("Best Perror thresh:", best_Perror_thresh)
