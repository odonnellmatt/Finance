# Portfolio Optimization: Efficient Frontier & Sharpe Ratio using Python

This project implements a **Monte Carlo simulation** to identify the **Efficient Frontier** and determine the optimal portfolio allocation for a given set of stocks.

By analyzing historical price data, the script generates thousands of random portfolio combinations to find the allocation that offers the highest risk-adjusted return (Maximum Sharpe Ratio).

## Features  
* **Data Acquisition:** Automated fetching of historical stock data using `yfinance`.  
* **Data Processing:** Calculation of daily log returns to normalize volatility analysis.  
* **Simulation:** Generation of 1,000,000 random portfolio combinations (Monte Carlo method).  
* **Optimization:** Identification of the "Maximum Sharpe Ratio" portfolio.  
* **Visualization:** Plotting the Efficient Frontier with `matplotlib`.

## 🛠 Dependencies  
To run this notebook, you will need the following Python libraries:

```bash  
pip install yfinance pandas numpy matplotlib
```

## **Usage & Code Explanation**

### **1. Import Libraries**

Standard financial analysis and visualization libraries are used.
```bash
import yfinance as yf  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
```

### **2. Select Tickers & Fetch Data**

We define the portfolio assets.
**Note:** If using yfinance for non-US stocks (like Australian stocks), append the exchange code (e.g. .AX for ASX).
```bash
# Select tickers, place into list 
tickers = ['BHP', 'AAPL', 'CBA.AX', 'MSFT', 'META', 'TSLA']

# Download stock data  
# We only need the 'Adj Close' column as it accounts for dividends and splits  
data = yf.download(tickers, start='2021-1-1', auto_adjust=False, multi_level_index=False) 

# Preview the data  
data.tail(10)
```

### **3. Prepare the Dataset (Log Returns)**

We calculate **Log Returns** rather than simple percent changes. Log returns are time-additive and generally preferable for mathematical modeling of volatility.
```bash
# Calculate log returns for each Adj Close  
log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1)) # shift(1) is just the value before
print(f"Data points before cleaning: {len(log_returns)}")

# Drop null values to avoid calculation errors  
log_returns = log_returns.dropna()

number_of_assets = len(data['Adj Close'].columns)  
print(f"Data points after cleaning (na removal): {len(log_returns)}")
```
### **4. Monte Carlo Simulation**

We iterate through a loop (default: 1,000,000 iterations) to generate random weightings for the assets. For every iteration, we calculate:

1. **Return:** Annualized expected return.  
2. **Volatility:** Annualized standard deviation (Risk).
```bash
# Initialize lists to store simulation results  
portfolio_returns = []  
portfolio_volatility = []  
portfolio_weights = []

# Run simulation  
# Note: Higher range values increase accuracy but require more compute time  
for x in range(1000000):  
    weights = np.random.random(number_of_assets)  
    weights /= np.sum(weights)  
    portfolio_weights.append(weights)

    # Annualize returns (252 trading days)  
    portfolio_returns.append(np.sum(weights * log_returns.mean() * 252))  
      
    # Calculate portfolio variance and convert to standard deviation (volatility)  
    portfolio_volatility.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights))))

# Convert lists to arrays for vector operations  
portfolio_returns = np.array(portfolio_returns)  
portfolio_volatility = np.array(portfolio_volatility)  
portfolio_weights = np.array(portfolio_weights)
```
### **5. Identify the Efficient Portfolio**

We calculate the **Sharpe Ratio** (Return / Volatility) for every simulated portfolio. The portfolio with the highest ratio is considered the "Efficient Portfolio."
```bash
# Calculate Sharpe Ratios  
sharpe_ratios = portfolio_returns / portfolio_volatility

# Find index of the highest Sharpe Ratio  
max_sr_index = np.argmax(sharpe_ratios)

# Retrieve metrics for the best portfolio  
max_sr_ret = portfolio_returns[max_sr_index]  
max_sr_vol = portfolio_volatility[max_sr_index]  
max_sr_weights = portfolio_weights[max_sr_index]

# Output Results  
print(f"Most Efficient Portfolio (Max Sharpe Ratio: {sharpe_ratios[max_sr_index]:.2f})")  
print(f"Annualised Return: {max_sr_ret:.2%}")  
print(f"Annualised Volatility: {max_sr_vol:.2%}")  
print("-" * 40)  
print("Portfolio Allocation:")  
for ticker, weight in zip(tickers, max_sr_weights):  
    print(f"{ticker}: {weight:.2%}")
```
### **6. Visualization**

Finally, we plot all simulated portfolios.

* **X-Axis:** Volatility (Risk)  
* **Y-Axis:** Returns  
* **Color Scale:** Sharpe Ratio  
* **Red Star:** The Optimal Portfolio
```bash
plt.figure(figsize=(12, 6))

# Scatter plot of all simulated portfolios  
scatter = plt.scatter(portfolio_volatility, portfolio_returns, c=sharpe_ratios, cmap='viridis', marker='o', s=3, alpha=0.5)  
plt.colorbar(scatter, label='Sharpe Ratio')

# Highlight the Optimal Portfolio  
plt.scatter(max_sr_vol, max_sr_ret, c='darkred', marker='*', s=50, label='Max Sharpe Ratio')

plt.xlabel('Expected Volatility (Risk)')  
plt.ylabel('Expected Return')  
plt.title(f'Efficient Frontier: {", ".join(tickers)}')  
plt.legend()  
plt.grid(True, linestyle='--', alpha=0.6)  
plt.show()
```
<img width="945" height="545" alt="output" src="https://github.com/user-attachments/assets/c942eb53-cb3c-485e-8d46-f3a9ed14098f" />

*Note: The Max Sharpe Ratio point is the Efficient Portfolio. This portfolio represents the highest risk-based return for this given dataset.*

---

***Disclaimer: This project is for educational purposes only and does not constitute financial advice.***

---
