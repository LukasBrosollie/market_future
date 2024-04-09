import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tensorflow.keras.layers import LSTM, Conv1D, GRU, Activation, Dense, Dropout, Bidirectional, concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Attention



def lr_reducer(monitor='val_loss', patience=15):
    reduce_lr = ReduceLROnPlateau(monitor=monitor, 
                                factor=0.25, 
                                cooldown=0,
                                patience=patience//5, 
                                verbose=1, 
                                min_lr=0.5e-6)
    return reduce_lr



def early_stop(monitor='val_loss', patience=15):
    early_stopping = EarlyStopping(monitor=monitor, 
                                patience=patience)
    return early_stopping




def training_milestones(output, monitor='val_loss'):
    checkpoint = ModelCheckpoint(filepath=output,
                                    monitor=monitor, 
                                    mode='auto',
                                    verbose=1,
                                    save_best_only=True)
    return checkpoint



def get_label(seq):
    diff = seq[-1] - seq[-2]
    if diff > 0:
        l = 0     # up
    else:
        l = 1     # Net
    
    return l



class ModelIndicators():
    
    def __init__(self,
                 N = 28,
                 cnn_layers = 1,
                 cnn_neurons = 32,
                 cnn_dropout = 0.2,
                 lstm_layers = 3,
                 dense_layers = 0,
                 lstm_dropout = 0.0,
                 dense_dropout = 0.5,
                 padding = 'same',
                 lstm_neurons = 20,
                 dense_neurons = 20,
                 activation = 'relu',
                 optimizer = Adam,
                 loss = 'binary_crossentropy',
                 learning_rate = 0.001
                 ):
        
        self.N = N
        self.cnn_layers = cnn_layers
        self.cnn_neurons = cnn_neurons
        self.cnn_dropout = cnn_dropout
        self.lstm_layers = lstm_layers
        self.dense_layers = dense_layers
        self.lstm_dropout = lstm_dropout
        self.dense_dropout = dense_dropout
        self.padding = padding
        self.lstm_neurons = lstm_neurons
        self.dense_neurons = dense_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate

    def __call__(self, oclh, macd, rsi, ema):

        x = oclh
        for i in range(self.cnn_layers):
            x = Conv1D(self.cnn_neurons[i], 3, padding=self.padding, activation=self.activation)(x)
            x = Dropout(self.cnn_dropout)(x)
        
        # LSTM Layers
        x_oclh = x
        for i in range(self.lstm_layers):
            if i < self.lstm_layers - 1:
                x_oclh = Bidirectional(GRU(self.lstm_neurons[i], activation=self.activation, dropout=self.lstm_dropout, 
                                       recurrent_dropout=self.lstm_dropout, return_sequences=True))(x_oclh)
            else:
                x_oclh = LSTM(self.lstm_neurons[i], activation=self.activation, dropout=self.lstm_dropout, 
                                        recurrent_dropout=self.lstm_dropout, return_sequences=False)(x_oclh)  

        x_macd = macd
        for i in range(self.lstm_layers):
            if i < self.lstm_layers - 1:
                x_macd = Bidirectional(GRU(self.lstm_neurons[i], activation=self.activation, dropout=self.lstm_dropout, 
                                       recurrent_dropout=self.lstm_dropout, return_sequences=True))(x_macd)
            else:
                x_macd = LSTM(self.lstm_neurons[i], activation=self.activation, dropout=self.lstm_dropout, 
                                        recurrent_dropout=self.lstm_dropout, return_sequences=False)(x_macd) 


        x_rsi = rsi
        for i in range(self.lstm_layers):
            if i < self.lstm_layers - 1:
                x_rsi = Bidirectional(GRU(self.lstm_neurons[i], activation=self.activation, dropout=self.lstm_dropout, 
                                       recurrent_dropout=self.lstm_dropout, return_sequences=True))(x_rsi)
            else:
                x_rsi = LSTM(self.lstm_neurons[i], activation=self.activation, dropout=self.lstm_dropout, 
                                        recurrent_dropout=self.lstm_dropout, return_sequences=False)(x_rsi) 

        x_ema = ema
        for i in range(self.lstm_layers):
            if i < self.lstm_layers - 1:
                x_ema = Bidirectional(GRU(self.lstm_neurons[i], activation=self.activation, dropout=self.lstm_dropout, 
                                       recurrent_dropout=self.lstm_dropout, return_sequences=True))(x_ema)
            else:
                x_ema = LSTM(self.lstm_neurons[i], activation=self.activation, dropout=self.lstm_dropout, 
                                        recurrent_dropout=self.lstm_dropout, return_sequences=False)(x_ema) 
                

        x = concatenate([Flatten()(x_oclh), Flatten()(x_macd), Flatten()(x_rsi), Flatten()(x_ema)])

        for i in range(self.dense_layers):
            if i < self.dense_layers - 1:
                x = Dense(self.dense_neurons[i], activation=self.activation)(x)
                x = Dropout(self.dense_dropout)(x)
            else:
                x = Dense(1)(x)

        o = Activation('sigmoid', name='output_layer')(x)


        model = Model(inputs=[oclh, macd, rsi, ema], outputs=o)
        opt_egine = self.optimizer(learning_rate = self.learning_rate)
        model.compile(optimizer=opt_egine, loss=self.loss)


        return model



