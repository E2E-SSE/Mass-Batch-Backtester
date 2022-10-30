# Mass Batch-Backtester
## Conduct large scale technical analysis using python

## The strategies include

- TREND
  - Simple Moving Average Crossover
  - Exponential Moving Average Crossover
  - Double Exponential Moving Average Crossover


- MOMENTUM
  - Relative Strength Index (RSI)
  - Money Flow Index (MFI)
  

- VOLATILITY
  - Bollinger Bands

___
## Dependencies
- [matplotlib](https://pypi.org/project/matplotlib/)
- [yfinance](https://pypi.org/project/yfinance/) 
- [pandas](https://pypi.org/project/pandas/)
- [numpy](https://pypi.org/project/numpy/)
- [ta](https://pypi.org/project/ta/)
___
## Setup
```bash
python -m pip install -r requirements.txt
```
___
After installing dependencies, open parameters.py and pass your own custom parameters for the backtesting
___
## parameters.py
```python

# Sample List

token_list = ["HEX-USD", "ATOM-USD", "LINK-USD", "FTT-USD", "XLM-USD", "ALGO-USD",
              "ICP-USD", "ETH-USD", "BTC-USD", "XMR-USD", "NEAR-USD", "KCS-USD",
              "BVOL-USD", "IBVOL-USD", "ETHBULL-USD", "ETHBEAR-USD", "SHIB-USD",
              "DOGE-USD", "MATIC-USD", "ETC-USD", "BNB-USD", "WBTC-USD", "WETH-USD",
              "BNBBULL-USD"]

mav1 = 13  # Moving average (FAST/SHORT)
mav2 = 34  # Moving average (SLOW/LONG)

stop_loss = 0.2
take_profit = 0.4  # As a percent

bband_period = 20
rsi_mfi_low = 20
initial_investment = 100000

```
___
## Startup
Run main.py and the script will start making directories and filling them.
