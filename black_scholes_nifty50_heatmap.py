import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma **2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma **2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Parameters
S = 19000         # Nifty 50 Spot Price
r = 0.07          # Risk-Free Rate (7%)
sigma = 0.12      # Volatility (12%)
T_days = 30       # Time to expiry in days
T = T_days / 365  # Convert days to years

strike_prices = np.arange(S-500, S+501, 100)      # Strike price from S-500 to S+500 in step of 100
volatility_range = np.linspace(0.08, 0.25, 10)    # Volatility from 8% to 25%

# Heatmap for call option values
call_matrix = np.zeros((len(volatility_range), len(strike_prices)))
for i, v in enumerate(volatility_range):
    for j, k in enumerate(strike_prices):
        call_matrix[i, j] = black_scholes_call(S, k, T, r, v)

plt.figure(figsize=(10, 6))
plt.imshow(call_matrix, cmap='viridis', aspect='auto', origin='lower',
           extent=[strike_prices[0], strike_prices[-1], volatility_range[0], volatility_range[-1]])
plt.colorbar(label='Call Option Value')
plt.xlabel('Strike Price')
plt.ylabel('Volatility')
plt.title('Nifty 50 Call Option Price Heatmap (Black-Scholes)')
plt.xticks(strike_prices)
plt.show()

