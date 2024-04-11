import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tensorflow.keras.layers import LSTM, Layer, Conv1D, GRU, Softmax, Activation, Dense, Dropout, BatchNormalization, Bidirectional, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import keras.backend as K
import glob



def normalize(data, norm_mode="max"):  

    if norm_mode == 'max':
        max = np.max(data)
        mean = np.mean(data)  
        if max == 0: max = 1
        data = (data - mean) / max
    elif norm_mode == 'std':
        std = np.std(data)
        mean = np.mean(data)  
        if std == 0: std = 1
        data = (data - mean) / std     
    else:
        raise NotImplementedError(f'No implementation for {norm_mode} normalization')   

    return [round(x, 3) for x in data.tolist()]  



def denormalize(data, normed_data, norm_mode="max"):  
    print(data)
    if norm_mode == 'max':
        max = np.max(data)
        mean = np.mean(data)  
        if max == 0: max = 1
        orig_data = (normed_data * max) + mean
    elif norm_mode == 'std':
        std = np.std(data)
        mean = np.mean(data)  
        if std == 0: std = 1
        orig_data = (normed_data * std) + mean  
    else:
        raise NotImplementedError(f'No implementation for {norm_mode} normalization')   

    return round(orig_data, 3)


def get_label(seq):
    diff = seq[-1] - seq[-2]
    if diff > 0:
        l = 0     # up
    elif diff < 0:
        l = 1     # Down
    else:
        l = 2     # Net
    
    return l



class BuildDataset():

    def __init__(self,
            N=10,
            L=1,
            F=4):
        
        self.N = N
        self.L = L
        self.F = F

    def __call__(self, dir):
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        y = []
        j=0      

        for crpfile in glob.glob(f'{dir}/*.csv'):
                data = pd.read_csv(crpfile)
                open = data['open'].tolist()
                close = data['close'].tolist()
                high = data['high'].tolist()
                low = data['low'].tolist()

                for i in range(len(data) - self.N - self.L + 1):
                    f1 = open[i:self.N+i].copy()
                    f2 = close[i:self.N+self.L+i].copy()     # target feature: includes the label
                    f3 = high[i:self.N+i].copy()
                    f4 = low[i:self.N+i].copy()

                    features = normalize(np.concatenate((f1, f2, f3, f4)), self.norm_mode)
                    assert len(features) == self.F * self.N + self.L

                    x1.append(features[:self.N])
                    x2.append(features[self.N:(2*self.N)])
                    x3.append(features[(2*self.N)+self.L:(3*self.N)+self.L])
                    x4.append(features[(3*self.N)+self.L:])
                    y.append([features[(2*self.N)]])
                    j+=1
        data = pd.DataFrame({'f1': x1, 'f2': x2, 'f3': x3, 
                                'f4': x4, 'labels': y})
        print(f'number of training samples: {j}')
        return data


class BuildDatasetClassification():

    def __init__(self,
            norm_mode='max',
            N=10,
            L=1,
            F=4):
        
        self.N = N
        self.L = L
        self.F = F
        self.norm_mode = norm_mode
    

    def __call__(self, dir):
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        y = []
        j=0      

        for crpfile in glob.glob(f'{dir}/*.csv'):
                data = pd.read_csv(crpfile)
                open = data['open'].tolist()
                close = data['close'].tolist()
                high = data['high'].tolist()
                low = data['low'].tolist()

                for i in range(len(data) - self.N - self.L + 1):
                    f1 = open[i:self.N+i].copy()
                    f2 = close[i:self.N+i].copy()
                    f3 = high[i:self.N+i].copy()
                    f4 = low[i:self.N+i].copy()

                    fx = close[i:self.N+self.L+i].copy()    
                    lbl = get_label(fx)

                    features = normalize(np.concatenate((f1, f2, f3, f4)), self.norm_mode)
                    assert len(features) == self.F * self.N 

                    x1.append(features[:self.N])
                    x2.append(features[self.N:2*self.N])
                    x3.append(features[2*self.N:3*self.N])
                    x4.append(features[3*self.N:])
                    y.append(lbl)

                    print(x1[-1])
                    print(x2[-1])
                    print(x3[-1])
                    print(x4[-1])
                    print(y[-1])

                    j+=1
        data = pd.DataFrame({'f1': x1, 'f2': x2, 'f3': x3, 
                                'f4': x4, 'labels': y})
        print(f'number of training samples: {j}')
        return data
        

class attention(Layer):

    def __init__(self, **kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), 
                               initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)   
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)

        return context


def lr_reducer(monitor='val_loss', patience=15):
    reduce_lr = ReduceLROnPlateau(monitor=monitor, 
                                factor=0.1, 
                                cooldown=0,
                                patience=patience-3, 
                                verbose=1, 
                                min_lr=0.5e-6)
    return reduce_lr



def early_stop(monitor='val_loss', patience=15):
    early_stopping = EarlyStopping(monitor=monitor, 
                                patience=patience)
    return early_stopping


class LearningRateMaster():

    def __init__(self, lr = 0.001):
        self.lr = lr

    def _lr_schedule(self, epoch): 
        'From https://github.com/smousavi05/EQTransformer'

        lrt = self.lr
        if epoch > 90:
            lrt *= 0.5e-2
        elif epoch > 60:
            lrt *= 1e-2
        elif epoch > 40:
            lrt *= 1e-1
        elif epoch > 20:
            lrt *= 1e-1
        print('Learning rate: ', lrt)

        return lrt


    def __call__(self):
        lr_scheduler = LearningRateScheduler(self._lr_schedule)

        return lr_scheduler


def training_milestones(output, monitor='val_loss'):
    checkpoint = ModelCheckpoint(filepath=output,
                                    monitor=monitor, 
                                    mode='auto',
                                    verbose=1,
                                    save_best_only=True)
    return checkpoint


class BuildModel():
    
    def __init__(self,
                 N = 10,
                 L = 1,
                 lstm_layers = 2,
                 dense_layers = 2,
                 lstm_dropout = 0.0,
                 dense_dropout = 0.5,
                 lstm_neurons = 20,
                 dense_neurons = 20,
                 activation = 'tanh',
                 optimizer = Adam,
                 loss = 'mean_squared_error',
                 epochs = 150,
                 batch_size = 64,
                 learning_rate = 0.001,
                 monitor = 'val_loss',
                 patience = 15
                 ):
        
        self.N = N
        self.L = L
        self.type = type
        self.lstm_layers = lstm_layers
        self.dense_layers = dense_layers
        self.lstm_dropout = lstm_dropout
        self.dense_dropout = dense_dropout
        self.lstm_neurons = lstm_neurons
        self.dense_neurons = dense_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.monitor = monitor
        self.patience = patience


    def __call__(self, inp):

        x = inp
        for i in range(self.lstm_layers):
            if i < self.lstm_layers - 1:
                x = Bidirectional(LSTM(self.lstm_neurons, activation=self.activation, dropout=self.lstm_dropout, 
                                       recurrent_dropout=self.lstm_dropout, return_sequences=True))(x)
            else:
                x = Bidirectional(GRU(self.lstm_neurons, activation=self.activation, dropout=self.lstm_dropout, 
                                       recurrent_dropout=self.lstm_dropout, return_sequences=False))(x)               

        for i in range(self.dense_layers):
            if i < self.dense_layers - 1:
                x = Dense(self.dense_neurons, activation=self.activation)(x)
                x = Dropout(self.dense_dropout)(x)
            else:
                x = Dense(self.L)(x)
        
        o = Activation('linear', name='output_layer')(x)


        model = Model(inputs=inp, outputs=o)
        opt_egine = self.optimizer(learning_rate = self.learning_rate)
        model.compile(optimizer=opt_egine, loss=self.loss)


        return model


class BuildModelClassification():
    
    def __init__(self,
                 N = 28,
                 cnn_layers = 2,
                 lstm_layers = 3,
                 dense_layers = 0,
                 cnn_dropout = 0.2,
                 lstm_dropout = 0.0,
                 dense_dropout = 0.5,
                 cnn_filters = [64, 32],
                 padding = 'same',
                 lstm_neurons = 20,
                 dense_neurons = 20,
                 filter_size = 3,
                 activation = 'relu',
                 optimizer = Adam,
                 loss = 'binary_crossentropy',
                 epochs = 150,
                 batch_size = 64,
                 learning_rate = 0.001,
                 monitor = 'val_loss',
                 patience = 15
                 ):
        
        self.N = N
        self.cnn_layers = cnn_layers
        self.lstm_layers = lstm_layers
        self.dense_layers = dense_layers
        self.cnn_dropout = cnn_dropout
        self.lstm_dropout = lstm_dropout
        self.dense_dropout = dense_dropout
        self.cnn_filters = cnn_filters
        self.padding = padding
        self.lstm_neurons = lstm_neurons
        self.dense_neurons = dense_neurons
        self.filter_size = filter_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.monitor = monitor
        self.patience = patience


    def __call__(self, inp):

        x = inp
        for i in range(self.cnn_layers):
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            x = Conv1D(self.cnn_filters[i], self.filter_size, padding=self.padding, activation=self.activation)(x)
            x = Dropout(self.cnn_dropout)(x)
            x = MaxPooling1D(2, padding=self.padding)(x)

        for i in range(self.lstm_layers):
            if i < self.lstm_layers - 1:
                x = Bidirectional(LSTM(self.lstm_neurons, activation=self.activation, dropout=self.lstm_dropout, 
                                       recurrent_dropout=self.lstm_dropout, return_sequences=True))(x)
            else:
                x = Bidirectional(GRU(self.lstm_neurons, activation=self.activation, dropout=self.lstm_dropout, 
                                        recurrent_dropout=self.lstm_dropout, return_sequences=False))(x)                     
          
        for i in range(self.dense_layers):
            if i < self.dense_layers - 1:
                x = Dense(self.dense_neurons, activation=self.activation)(x)
                x = Dropout(self.dense_dropout)(x)
            else:
                x = Dense(3)(x)

        o = Activation('softmax', name='output_layer')(x)


        model = Model(inputs=inp, outputs=o)
        opt_egine = self.optimizer(learning_rate = self.learning_rate)
        model.compile(optimizer=opt_egine, loss=self.loss)


        return model
