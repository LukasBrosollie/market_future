import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, LeakyReLU, GRU, Activation, Input, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.models import Model
from obspy.signal.cross_correlation import correlate
import os, glob

data_dir = '../original_data'
N = 10                    # inputs length 
L = 1                     # label length


# Model parameters
lr = 0.001
n = 4       # number of features
ep = 200
bs = 64
pt = 30       

dataset = f'crypto_N{N}_L{L}_allfeatures.csv'
outmodel = f'model_BEST_tanh.h5'

def normalize(data):  
    data = data - np.mean(data)
    max_data = float(np.max(data))
    if max_data == 0:
        max_data = 1 
    data = data / max_data 
    return data.tolist()

def _lr_schedule(epoch):
    'From https://github.com/smousavi05/EQTransformer' 

    lrt = lr
    if epoch > 90:
        lrt *= 0.5e-3
    elif epoch > 60:
        lrt *= 1e-3
    elif epoch > 40:
        lrt *= 1e-2
    elif epoch > 20:
        lrt *= 1e-1
    print('Learning rate: ', lrt)
    return lrt


x1 = []
x2 = []
x3 = []
x4 = []
y = []
j=0


if not os.path.isfile(dataset):
    for crpfile in glob.glob(f'{data_dir}/*.csv'):
        data = pd.read_csv(crpfile)
        open = data['open'].tolist()
        close = data['close'].tolist()
        high = data['high'].tolist()
        low = data['low'].tolist()

            
        for i in range(len(data) - N - L + 1):
            f1 = open[i:N+i].copy()
            f2 = close[i:N+L+i].copy()     # target feature: includes the label
            f3 = high[i:N+i].copy()
            f4 = low[i:N+i].copy()

            features = normalize(np.concatenate((f1, f2, f3, f4)))
            '''
            N=10, L=1
            aaaaaaaaaa
            bbbbbbbbbbb
            cccccccccc
            dddddddddd
            # concatenate:
                                2N+L               4N
                                 |                  |
            aaaaaaaaaabbbbbbbbbbbccccccccccdddddddddd
            |         |         |          |
    indx    0         N        2N(lbl)    3N+L
            '''
            assert len(features) == n * N + L

            x1.append(features[:N])
            x2.append(features[N:(2*N)])
            x3.append(features[(2*N)+L:(3*N)+L])
            x4.append(features[(3*N)+L:])
            y.append([features[(2*N)]])
            j+=1

    data = pd.DataFrame({'f1': x1, 'f2': x2, 'f3': x3, 
                         'f4': x4, 'labels': y})
    data.to_csv(dataset, index=False)
    print(f'number of training samples: {j}')


data = pd.read_csv(dataset)

inputs_1 = np.array([eval(seq) for seq in data['f1'].tolist()])
inputs_2 = np.array([eval(seq) for seq in data['f2'].tolist()])
inputs_3 = np.array([eval(seq) for seq in data['f3'].tolist()])
inputs_4 = np.array([eval(seq) for seq in data['f4'].tolist()])
labels = np.array([eval(seq) for seq in data['labels'].tolist()])

input_data = np.stack((inputs_1, inputs_2, inputs_3, inputs_4), axis=-1)
assert input_data.shape == (len(data), N, n), f'input data has a shape of {input_data.shape}'

i_tr, i_te, l_tr, l_te = train_test_split(
    input_data, 
    labels, 
    test_size=0.2, 
    shuffle=True,
    random_state=42
)


input_layer = Input(shape=(N, n), name='input_layer')
x = Bidirectional(LSTM(20, activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=True))(input_layer)
x = Bidirectional(GRU(20, activation='tanh', dropout=0.0, recurrent_dropout=0.0, return_sequences=False))(x) 
x = Dense(2*N, activation='tanh')(x)
x = Dropout(0.5)(x)
x = Dense(L)(x)
o = Activation('linear', name='output_layer')(x)
model = Model(inputs=input_layer, outputs=o)


reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1, 
                              cooldown=0,
                              patience=pt-3, 
                              verbose=1, 
                              min_lr=0.5e-6)

early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=pt)

lr_scheduler = LearningRateScheduler(_lr_schedule)

checkpoint = ModelCheckpoint(filepath=outmodel,
                                monitor='val_loss', 
                                mode='auto',
                                verbose=1,
                                save_best_only=True)

optimizer = Adam(learning_rate = lr)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()
history = model.fit(i_tr, 
                    l_tr, 
                    epochs=ep, 
                    batch_size=bs, 
                    validation_data=(i_te, l_te), 
                    callbacks=[checkpoint, reduce_lr, lr_scheduler, early_stopping])


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Performance')
plt.legend()
plt.savefig(f'{dataset.replace(".csv", ".jpg")}', dpi = 400)
plt.show()
