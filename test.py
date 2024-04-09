import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from tensorflow.keras.models import load_model
import yfinance as yf



N = 7
mode = 'split'
model = load_model('./models_regression/Reg_MSE_0.000119_MAE_0.0067/crypto_indicators_2.h5')
ticker="BTC-USD"


scaler = MinMaxScaler(feature_range=(0, 1))
msft = yf.Ticker(ticker)
df = msft.history(start="2023-10-01", end="2023-12-14", interval="1d")
df = df.dropna()
df = df.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)
df.columns = ['open', 'high', 'low', 'close']
print(df)
print(len(df))


# generate the indicators
df.ta.macd(append=True, talib=False)
df.ta.rsi(length=14, append=True, talib=False)
df.ta.stoch(append=True, talib=False)
df.ta.bbands(length=20, append=True, talib=False)
df.ta.ema(length=9, append=True, talib=False)
df.ta.ema(length=13, append=True, talib=False)
df.ta.ema(length=20, append=True, talib=False)
df = df.dropna()
print(df)
print(len(df))
data_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)



x_o, x_c, x_h, x_l = [], [], [], []
x_mac1, x_mac2, x_mac3 = [], [], []
x_rsi = []
x_stc1, x_stc2 = [], []
x_bb1, x_bb2, x_bb3, x_bb4, x_bb5 = [], [], [], [], []
x_ema1, x_ema2, x_ema3 = [], [], []
lbls = []


for i in range(len(data_scaled) - N-1):
    tmp = data_scaled[i:N+i+1]
    lbl = tmp['close'].tolist()[-1]
    batch = tmp[:-1]
    
    x_o.append(np.array(batch['open'].tolist()))
    x_c.append(np.array(batch['close'].tolist()))
    x_h.append(np.array(batch['high'].tolist()))
    x_l.append(np.array(batch['low'].tolist()))
    x_mac1.append(np.array(batch['MACD_12_26_9'].tolist()))
    x_mac2.append(np.array(batch['MACDh_12_26_9'].tolist()))
    x_mac3.append(np.array(batch['MACDs_12_26_9'].tolist()))
    x_rsi.append(np.array(batch['RSI_14'].tolist()))
    x_bb1.append(np.array(batch['BBL_20_2.0'].tolist()))
    x_bb2.append(np.array(batch['BBM_20_2.0'].tolist()))
    x_bb3.append(np.array(batch['BBU_20_2.0'].tolist()))
    x_bb4.append(np.array(batch['BBB_20_2.0'].tolist()))
    x_bb5.append(np.array(batch['BBP_20_2.0'].tolist()))
    x_stc1.append(np.array(batch['STOCHk_14_3_3'].tolist()))
    x_stc2.append(np.array(batch['STOCHd_14_3_3'].tolist()))
    x_ema1.append(np.array(batch['EMA_9'].tolist()))
    x_ema2.append(np.array(batch['EMA_13'].tolist()))
    x_ema3.append(np.array(batch['EMA_20'].tolist()))
    lbls.append(lbl)

if not mode == 'total':
    oclh = np.stack((x_o,x_c,x_l,x_h), axis=-1)
    macd = np.stack((x_mac1,x_mac2,x_mac3), axis=-1)
    rsi = x_rsi
    ema = np.stack((x_ema1,x_ema2,x_ema3), axis=-1)
    bb = np.stack((x_bb1,x_bb2,x_bb3,x_bb4,x_bb5), axis=-1)
    stc = np.stack((x_stc1,x_stc2), axis=-1)
    lbls = np.array(lbls)

    preds = []
    for i in range(len(lbls)):
        val_inp = {'oclh':oclh[i].reshape(-1,N,4), 'macd':macd[i].reshape(-1,N,3), 
                   'rsi':rsi[i].reshape(-1,N,1), 'ema': ema[i].reshape(-1,N,3),
                  'bb': bb[i].reshape(-1,N,5), 'stc':stc[i].reshape(-1,N,2)}
        p = model.predict(val_inp)[0][0]
        preds.append(p)
else:
    input_total = np.stack((x_o,x_c,x_l,x_h,x_mac1,x_mac2,x_mac3,x_rsi,x_ema1,x_ema2,x_ema3,
                        x_bb1,x_bb2,x_bb3,x_bb4,x_bb5,x_stc1,x_stc2), axis=-1)
    preds = []
    for i in range(len(lbls)):
        p = model.predict(input_total[i].reshape(-1,N,18))[0][0]
        preds.append(p) 


fig, ax = plt.subplots(figsize=(12,8))
ax.plot(lbls, linestyle='-', label='Actual Prices')
ax.plot(preds, linestyle='-', label='Prediction')
plt.xticks(rotation=45)
ax.set_xlabel('Day')
ax.set_ylabel('Price')
ax.set_title('Crypto Prediction Results')
plt.legend()
plt.savefig('regression_test.jpg', dpi=500)
plt.show()