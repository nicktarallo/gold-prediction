import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import simulate_gains

threshold = 0
best_length = 0
max_end_result = 0
max_window_length = 20

for window_length in range(1, max_window_length + 1):
    print("WINDOW LENGTH:", window_length)
    df = pd.read_csv('AllGoldDataDataPrice.csv', parse_dates=['Date'], index_col=['Date'])
    df.dropna(inplace=True)

    moving_avg = df.loc[:, "USD"].rolling(window=window_length).mean().iloc[window_length-1:].values

    df = df.iloc[window_length-1:]
    df["MOVING_AVG"] = moving_avg
    df = df.iloc[max_window_length-window_length:]
    #df = df.iloc[:len(df["MOVING_AVG"])//2]

    df['INCREASING'] = (df['MOVING_AVG'] - df['MOVING_AVG'].shift() > threshold).fillna(0).astype(int)

    #df.loc[:, "USD":"MOVING_AVG"].plot(use_index=True)
    #plt.show()
    print(df)

    bot_values = simulate_gains.simulate_gains(df['USD'], df['INCREASING'])
    all_hold_values = simulate_gains.simulate_gains(df['USD'], np.ones(len(df['INCREASING'])))

    df["VALUE"] = bot_values
    df["ALLHOLD"] = all_hold_values

    # prices_test.plot(y="VALUE", use_index=True)
    df.loc[:, "VALUE":"ALLHOLD"].plot(use_index=True)

    plt.show()

    print("Model result:", bot_values[-1])
    print("All hold result:", all_hold_values[-1])
    if bot_values[-1] > max_end_result:
        max_end_result = bot_values[-1]
        best_length = window_length

print("Best window length:", best_length)
print("End value:", max_end_result)
