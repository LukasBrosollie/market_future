import yfinance as yf



def get_data(ticker):
  msft = yf.Ticker(ticker)
  df = msft.history(start="2023-11-14", end="2023-12-14", interval="1d")
  df = df.dropna()
  df = df.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)
  df.columns = ['open', 'high', 'low', 'close']

  return df

df = get_data("BTC-USD")
print(df)