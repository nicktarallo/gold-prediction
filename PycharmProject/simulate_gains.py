import numpy as np


def simulate_gains(prices, choices, initial=1000):
    n = len(prices)
    value = initial
    values = np.zeros(n)
    values[0] = value
    for i in range(1, n):
        hold = choices[i-1]
        if hold == 1:
            value = value * (1 + ((prices[i] - prices[i-1]) / prices[i-1]))
        values[i] = value

    return values
