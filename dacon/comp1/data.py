
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras import metrics
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, MaxoutDense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold, cross_validate

# 데이터 

train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)

print(train.shape)

train = train.drop('840_dst', axis=1)

print(train.shape)




# rho = train.sort_values(by =['rho'], axis=0)
# rho.to_csv('./data/dacon/comp1/rho.csv', index_label='id')

# sns.pairplot(train, diag_kind='hist')
# plt.show()

# corr_matrix = train.corr()
# hbb = corr_matrix["na"].sort_values(ascending=True)
# print(hbb)

# sns.pairplot(train_dst, diag_kind='hist')


# import missingno as msno
# msno.matrix(train_dst)  # dst가 난리다 
# plt.show()

# train.hist(bins=50, figsize=(20,20))


# plt.figure(figsize=(4, 12))
# sns.heatmap(train.corr().loc['rho':'990_dst', 'hhb':].abs())

# test.filter(regex='_dst$',axis=1).tail().T.plot() 
# plt.show()


# train = train.drop('840_dst', axis=1)
# test = test.drop(['810_dst','830_dst','840_dst'], axis=1 )



'''
class AutoEncoder:
    def __init__(self, encoding_dim):
        self.encoding_dim = encoding_dim
    def build_train_model(self, input_shape, encoded1_shape, encoded2_shape, decoded1_shape, decoded2_shape):
        input_data = Input(shape=(1, input_shape))
        encoded1 = Dense(encoded1_shape, activation="relu", activity_regularizer=regularizers.l2(0))(input_data)
        encoded2 = Dense(encoded2_shape, activation="relu", activity_regularizer=regularizers.l2(0))(encoded1)
        encoded3 = Dense(self.encoding_dim, activation="relu", activity_regularizer=regularizers.l2(0))(encoded2)
        decoded1 = Dense(decoded1_shape, activation="relu", activity_regularizer=regularizers.l2(0))(encoded3)
        decoded2 = Dense(decoded2_shape, activation="relu", activity_regularizer=regularizers.l2(0))(decoded1)
        decoded = Dense(input_shape, activation="sigmoid", activity_regularizer=regularizers.l2(0))(decoded2)
        autoencoder = Model(inputs=input_data, outputs=decoded)
        encoder = Model(input_data, encoded3)
        # Now train the model using data we already preprocessed
        autoencoder.compile(loss="mae", optimizer="adam")
        train = pd.read_csv("preprocessing/rbm_train.csv", index_col=0)
        ntrain = np.array(train)
        train_data = np.reshape(ntrain, (len(ntrain), 1, input_shape))
        # print(train_data)
        # autoencoder.summary()
        autoencoder.fit(train_data, train_data, epochs=1000)


class NeuralNetwork:
    def __init__(self, input_shape, stock_or_return):
        self.input_shape = input_shape
        self.stock_or_return = stock_or_return
    def make_train_model(self):
        input_data = Input(shape=(1, self.input_shape))
        lstm = LSTM(5, input_shape=(1, self.input_shape), return_sequences=True, activity_regularizer=regularizers.l2(0.003),
                       recurrent_regularizer=regularizers.l2(0), dropout=0.2, recurrent_dropout=0.2)(input_data)
        perc = Dense(5, activation="sigmoid", activity_regularizer=regularizers.l2(0.005))(lstm)
        lstm2 = LSTM(2, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
                        dropout=0.2, recurrent_dropout=0.2)(perc)
        out = Dense(1, activation="sigmoid", activity_regularizer=regularizers.l2(0.001))(lstm2)
        model = Model(input_data, out)
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
        # load data
        train = np.reshape(np.array(pd.read_csv("features/autoencoded_train_data.csv", index_col=0)),
                           (len(np.array(pd.read_csv("features/autoencoded_train_data.csv"))), 1, self.input_shape))
        train_y = np.array(pd.read_csv("features/autoencoded_train_y.csv", index_col=0))
        # train_stock = np.array(pd.read_csv("train_stock.csv"))
        # train model
        model.fit(train, train_y, epochs=2000)

'''